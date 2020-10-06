import argparse
import random
import itertools
import json
import torch
from rouge import Rouge

from summarizer.rnn.data import ReadData
from summarizer.rnn.model import EncoderRNN, DecoderRNN, AttnDecoderRNN1

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
    parser.add_argument('--max_input_length', dest='max_input_length', type=int, default=512,
                        help='Maximum input length for encoder')
    parser.add_argument('--max_output_length', dest='max_output_length', type=int, default=512,
                        help='Maximum output length for decoder')
    parser.add_argument('--dropout', dest='dropout', type=int, default=0.1,
                        help='Dropout')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=500,
                        help='LSTM hidden dimension')
    parser.add_argument('--encoder_n_layers', dest='encoder_n_layers', type=int, default=2,
                        help='Number of LSTM encoder layers')
    parser.add_argument('--decoder_n_layers', dest='decoder_n_layers', type=int, default=2,
                        help='Number of LSTM decoder layers')
    parser.add_argument('--with_attention', dest='with_attention', action='store_true',
                        default=False, help='Attention mechanism included or not?')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true',
                        default=False, help='Testing mode?')
    args = parser.parse_args()
    return args

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

def test(args, vocab_json, batches, encoder, decoder, device):
    encoder.eval()
    decoder.eval()
    SOS_token = 2

    pred_answers = []
    ref_answers = []
    #for i in range(args.data_size//args.batch_size):
    for i in range(len(batches)):
        input_variable, lengths, ref_output = batches[i]
        input_variable.to(device)
        lengths.to(device)

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(args.batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        answer = []
        for t in range(args.max_output_length):
            #decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
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
        ref_answers.append(ref_output[0])
        
    rouge = Rouge()
    #import IPython; IPython.embed(); exit(1)
    scores = rouge.get_scores(pred_answers, ref_answers, avg=True)

    print(scores)
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
        decoder = AttnDecoderRNN1(args.hidden_size, voc.num_words, args.decoder_n_layers, args.dropout)
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