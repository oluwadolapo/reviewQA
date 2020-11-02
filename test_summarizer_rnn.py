import argparse
import random
import itertools
import json
import torch
import nltk
import unicodedata
import re
from rouge import Rouge
import pandas as pd

#from summarizer.rnn.data import ReadData
from summarizer.rnn.model import EncoderRNN, DecoderRNN, AttnDecoderRNN1, AttnDecoderRNN2

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_encoder_path', dest='load_encoder_path', type=str, default=None,
                        help='Directory for loading encoder')
    parser.add_argument('--load_decoder_path', dest='load_decoder_path', type=str, default=None,
                        help='Directory for loading decoder')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Directory for loading testing data')
    parser.add_argument('--load_vocab_path', dest='load_vocab_path', type=str, default=None,
                        help='Directory for loading vocabulary')
    parser.add_argument('--max_input_length', dest='max_input_length', type=int, default=1000,
                        help='Maximum input length for encoder')
    parser.add_argument('--max_output_length', dest='max_output_length', type=int, default=512,
                        help='Maximum output length for decoder')
    parser.add_argument('--dropout', dest='dropout', type=int, default=0.1,
                        help='Dropout')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512,
                        help='LSTM hidden dimension')
    parser.add_argument('--encoder_n_layers', dest='encoder_n_layers', type=int, default=1,
                        help='Number of LSTM encoder layers')
    parser.add_argument('--decoder_n_layers', dest='decoder_n_layers', type=int, default=1,
                        help='Number of LSTM decoder layers')
    parser.add_argument('--with_attention', dest='with_attention', action='store_true',
                        default=False, help='Attention mechanism included or not?')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true',
                        default=False, help='Testing mode?')
    args = parser.parse_args()
    return args

class ReadData:
    def __init__(self, MAX_LENGTH):
        """
        MAX_LENGTH ==> Maximum sentence length to consider (max words)
        """
        self.MAX_LENGTH = MAX_LENGTH

    def unicodeToAscii(self, s):
        """
        Turns a Unicode string to plain ASCII
        """
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def normalizeString(self, s):
        """
        Convert to lowercase, trim white spaces, lines...etc, and remove
        non-letter characters.
        """
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])",r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def readFile(self, args):
        print()
        print("Reading and processing file...Please wait")
        df = pd.read_json(args.data_path, orient='split')
        #df = df.head(100)
        if not args.test_mode:
            df = df[:int(0.98*len(df))]
        
        questions = [q if q.endswith("?") else q+"?" for q in df.question]
        reviews = [r for r in df.passages]
        answers = [a for a in df.answers]

        questions = ["<s> " + q.lower() + " </s>" for q in questions]
        reviews = [[p.lower() for p in r] for r in reviews]
        answers = ["<s> " + a.lower() + " </s>" for a in answers]

        pairs = []
        for i, q in enumerate(questions):
            pair = []
            for r in reviews[i]:
                q = q + " " + r
            pair.append(q)
            pair.append(answers[i])
            pairs.append(pair)
        
        print("Done Reading!")
        print("size of remaining pairs: " + str(len(pairs)))
        return pairs

class PrepareData:
    def __init__(self):
        self.PAD_token = 0 # Used for padding short sentences <pad>
        self.UNK_token = 1 # unknown words token <unk>
        self.SOS_token = 2 # Start-of-sentence token <s>
        self.EOS_token = 3 # End-of-sentence token </s>

    def indexesFromSentence(self, vocab_json, sentence):
        indexes = []
        #for word in sentence.split(' '):
        for word in nltk.word_tokenize(sentence):
            try:
                indexes.append(vocab_json["voc"]["word2index"][word])
            except KeyError:
                indexes.append(1)
        #return [voc.word2index[word] for word in sentence.split(' ')]
        return indexes

    def zeroPadding(self, l, fillvalue = 0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))


    def inputVar(self, l, vocab_json):
        """
        Returns padded input sequence tensor and as well as a tensor
        of lengths for each of the sequences in the batch
        """
        indexes_batch = [self.indexesFromSentence(vocab_json, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths


    def batch2TestData(self, vocab_json, pair_batch):
        #Sort the questions in descending length
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch = []
        output_batch = []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, vocab_json)
        return inp, lengths, output_batch


def prepare_data(args):
    Data = ReadData(args.max_input_length)
    pairs = Data.readFile(args)
    with open(args.load_vocab_path, 'r') as json_data:
        vocab_json = json.load(json_data)

    pre_data = PrepareData()
    batches = []
    """
    steps = args.data_size//args.batch_size
    import IPython; IPython.embed(); exit(1)
    for i in range(steps):   
        batch = pre_data.batch2TestData(vocab_json, [random.choice(pairs)
          for _ in range(args.batch_size)])
        batches.append(batch)
    """
    for i in range(len(pairs)):
        batch = pre_data.batch2TestData(vocab_json, [pairs[i]])
        batches.append(batch)
    
    return vocab_json, batches

def create_mask(src):
    mask = (src != 0).permute(1, 0)
    return mask

def test(args, vocab_json, batches, encoder, decoder, device):
    encoder.eval()
    decoder.eval()
    SOS_token = 2

    df = pd.read_json(args.data_path, orient='split')
    #df = df[:1]
    #answers = [a for a in df.answers]
    multiple_answers = [a for a in df.multiple_answers]
    #answers = ["<s> " + a.lower() + " </s>" for a in answers]
    ref_answers = [["<s> " + a.lower() + " </s>" for a in ans] for ans in multiple_answers]

    pred_answers = []
    #ref_answers = []
    #for i in range(args.data_size//args.batch_size):
    for i in range(len(batches)):
        input_variable, lengths, ref_output = batches[i]
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)

        encoder_mask = create_mask(input_variable)
        encoder_mask = encoder_mask.to(device)

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(args.batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        #decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_hidden = encoder_hidden.unsqueeze(-1)
        decoder_hidden = decoder_hidden.transpose(2, 0)
        decoder_hidden = decoder_hidden.transpose(2, 1)

        answer = []
        for t in range(args.max_output_length):
            if args.with_attention:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, encoder_mask)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            answer.append(vocab_json["voc"]["index2word"][str(topi[0][0].item())])
            #if vocab_json["voc"]["index2word"][str(topi[0][0])] == "</s>":
            if topi[0][0].item() == 0 or topi[0][0].item() == 12:
                break
            decoder_input = torch.LongTensor([[topi[i][0] for i in range (args.batch_size)]])
            decoder_input = decoder_input.to(device)
        ans = "<s> "
        #for word in answer[1:-1]:
        for word in answer:
            ans += word + " "
        pred_answers.append(ans)
        #ref_answers.append(ref_output[0])
        
    
    rouge = Rouge()
    
    all_rouge_1p = []
    all_rouge_1r = []
    all_rouge_1f = []
    all_rouge_2p = []
    all_rouge_2r = []
    all_rouge_2f = []
    all_rouge_lp = []
    all_rouge_lr = []
    all_rouge_lf = []

    for count in range(len(pred_answers)):
        rouge_1p = []
        rouge_1r = []
        rouge_1f = []
        rouge_2p = []
        rouge_2r = []
        rouge_2f = []
        rouge_lp = []
        rouge_lr = []
        rouge_lf = []
        for ans in ref_answers[count]:
            hyp = []
            ref = []
            hyp.append(pred_answers[count])
            ref.append(ans)
            scores = rouge.get_scores(hyp, ref, avg=True)
            rouge_1p.append(scores['rouge-1']['p'])
            rouge_1r.append(scores['rouge-1']['r'])
            rouge_1f.append(scores['rouge-1']['f'])
            rouge_2p.append(scores['rouge-2']['p'])
            rouge_2r.append(scores['rouge-2']['r'])
            rouge_2f.append(scores['rouge-2']['f'])
            rouge_lp.append(scores['rouge-l']['p'])
            rouge_lr.append(scores['rouge-l']['r'])
            rouge_lf.append(scores['rouge-l']['f'])

        rouge_1p.sort()
        rouge_1r.sort()
        rouge_1f.sort()
        rouge_2p.sort()
        rouge_2r.sort()
        rouge_2f.sort()
        rouge_lp.sort()
        rouge_lr.sort()
        rouge_lf.sort()

        all_rouge_1p.append(rouge_1p[-1])
        all_rouge_1r.append(rouge_1r[-1])
        all_rouge_1f.append(rouge_1f[-1])
        all_rouge_2p.append(rouge_2p[-1])
        all_rouge_2r.append(rouge_2r[-1])
        all_rouge_2f.append(rouge_2f[-1])
        all_rouge_lp.append(rouge_lp[-1])
        all_rouge_lr.append(rouge_lr[-1])
        all_rouge_lf.append(rouge_lf[-1])
    
    print()
    print("rouge_1p:", sum(all_rouge_1p)/len(all_rouge_1p))
    print("rouge_1r:", sum(all_rouge_1r)/len(all_rouge_1r))
    print("rouge_1f:", sum(all_rouge_1f)/len(all_rouge_1f))
    print()
    print("rouge_2p:", sum(all_rouge_2p)/len(all_rouge_2p))
    print("rouge_2r:", sum(all_rouge_2r)/len(all_rouge_2r))
    print("rouge_2f:", sum(all_rouge_2f)/len(all_rouge_2f))
    print()
    print("rouge_lp:", sum(all_rouge_lp)/len(all_rouge_lp))
    print("rouge_lr:", sum(all_rouge_lr)/len(all_rouge_lr))
    print("rouge_lf:", sum(all_rouge_lf)/len(all_rouge_lf))

    #rouge = Rouge()
    #scores = rouge.get_scores(pred_answers, ref_answers, avg=True)

    #print(scores)
    #import IPython; IPython.embed(); exit(1)
    

def main():
    args = get_params()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")
    vocab_json, batches = prepare_data(args)

    ##### Define the encoder and decoder #####
    encoder = EncoderRNN(vocab_json["voc"]["num_words"], args.hidden_size, args.encoder_n_layers, args.dropout)
    
    if args.with_attention:
        decoder = AttnDecoderRNN2(args.hidden_size, vocab_json["voc"]["num_words"], args.decoder_n_layers, args.dropout)
    else:
        decoder = DecoderRNN(args.hidden_size, vocab_json["voc"]["num_words"], args.decoder_n_layers, args.dropout)
    
    encoder.load_state_dict(torch.load(args.load_encoder_path, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(args.load_decoder_path, map_location=torch.device('cpu')))
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    test(args, vocab_json, batches, encoder, decoder, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')