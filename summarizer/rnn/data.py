import os
import unicodedata
import re
import itertools
import pandas as pd
import torch

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

    def filterPair(self, p):
        """
        Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
        """
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split()) < self.MAX_LENGTH and len(p[1].split()) < self.MAX_LENGTH

    # Filter pairs using filterPair condition
    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def readFile(self, args):
        print()
        print("Reading and processing file...Please wait")
        df = pd.read_json(args.data_path, orient='split')
        df = df[:int(0.99*len(df))]
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
        
        # Split every line into pairs and normalize
        #pairs = [[self.normalizeString(sentence) for sentence in pair.split('\t')] for pair in lines]
        pairs = self.filterPairs(pairs)
        print("Done Reading!")
        print("size of remaining pairs: " + str(len(pairs)))
        return pairs


class Vocabulary:
    def __init__(self, name):
        self.PAD_token = 0 # Used for padding short sentences <pad>
        self.UNK_token = 1 # unknown words token <unk>
        self.SOS_token = 2 # Start-of-sentence token <s>
        self.EOS_token = 3 # End-of-sentence token </s>
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "<pad>", self.UNK_token: "<unk>", self.SOS_token: "<s>", self.EOS_token: "</s>"}
        self.num_words = 4 #Count SOS, EOS, PAD, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        """
        Remove words below a certain count threshold
        """
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words)/len(self.word2index)))
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "<pad>", self.UNK_token: "<unk>", self.SOS_token: "<s>", self.EOS_token: "</s>"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

    def filter_voc(self, pairs, MIN_COUNT = 3):
        """
        Filters out pairs with trimmed words
        """
        self.trim(MIN_COUNT)
        keep_pairs = []
        for pair in pairs:
          input_sentence = pair[0]
          output_sentence = pair[1]
          keep_input = True
          keep_output = True
          # Check input sentence
          for word in input_sentence.split(' '):
            if word not in self.word2index:
              keep_output = False
              break
          if keep_input and keep_output:
            keep_pairs.append(pair)
        return keep_pairs

class PrepareData:
    def __init__(self):
        self.PAD_token = 0 # Used for padding short sentences <pad>
        self.UNK_token = 1 # unknown words token <unk>
        self.SOS_token = 2 # Start-of-sentence token <s>
        self.EOS_token = 3 # End-of-sentence token </s>

    def indexesFromSentence(self, voc, sentence):
        indexes = []
        for word in sentence.split(' '):
            try:
                indexes.append(voc.word2index[word])
            except KeyError:
                indexes.append(1)
        #return [voc.word2index[word] for word in sentence.split(' ')]
        return indexes

    def zeroPadding(self, l, fillvalue = 0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(self, l, value=0):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def inputVar(self, l, voc):
        """
        Returns padded input sequence tensor and as well as a tensor
        of lengths for each of the sequences in the batch
        """
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    def outputVar(self, l, voc):
        """
        Returns padded target sequence tensor, padding mask, and max target length
        """
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        mask = self.binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        return padVar, mask, max_target_len

    def batch2TrainData(self, voc, pair_batch):
        """
        Returns all items for a given batch of pairs
        """
        #Sort the questions in descending length
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, voc)
        #assert len(inp) == lengths[0]
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len
