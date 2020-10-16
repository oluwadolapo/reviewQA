from torchtext import data, datasets
import torchtext.vocab as voc
from torchtext.vocab import GloVe, FastText, CharNGram
import torch

import random
import numpy as np

import sys

def train_data(args, device):
    if args.local_run:
        TEXT = data.Field(tokenize = 'spacy', include_lengths=True, fix_length=20)
        LABEL = data.LabelField(dtype = torch.float)
    else:
        TEXT = data.Field(tokenize = 'spacy', include_lengths=True)
        LABEL = data.LabelField(dtype = torch.float)

    fields = {'quest_rev': ('text', TEXT), 'is_answerable': ('label', LABEL)}
    train_data, valid_data = data.TabularDataset.splits(path = args.data_path,
                                    train = args.train_file_name,
                                    validation = args.val_file_name,
                                    format = 'json',
                                    fields = fields)

    #train_data, valid_data = train_data.split(random_state = random.seed(args.random_seed))

    if args.resume_training:
        custom_embeddings = voc.Vectors(name = args.load_emb_path,
                                  #cache = 'custom_embeddings',
                                  unk_init = torch.Tensor.normal_)
        TEXT.build_vocab(train_data, 
                 max_size = args.max_voc_size, 
                 vectors = custom_embeddings)

        LABEL.build_vocab(train_data)
    else:
        TEXT.build_vocab(train_data, max_size = args.max_voc_size, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

    # Save vocabulary
    with open(args.save_voc_path, 'w+') as f:
        for token, index in TEXT.vocab.stoi.items():
            f.write(f'{index}\t{token}\n')

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = args.batch_size, 
        sort_within_batch=True,
        sort_key = lambda x: x.text,
        device = device)
    
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    return train_iterator, valid_iterator, TEXT.vocab, UNK_IDX, PAD_IDX

def test_data(args, device):
    if args.local_run:
        TEXT = data.Field(tokenize = 'spacy', include_lengths=True, fix_length=20)
        LABEL = data.LabelField(dtype = torch.float)
    else:
        TEXT = data.Field(tokenize = 'spacy', include_lengths=True)
        LABEL = data.LabelField(dtype = torch.float)

    fields = {'quest_rev': ('text', TEXT), 'is_answerable': ('label', LABEL)}
    train_data, test_data = data.TabularDataset.splits(path = args.data_path,
                                    train = args.train_file_name,
                                    test = args.test_file_name,
                                    format = 'json',
                                    fields = fields)
    
    """
    custom_embeddings = voc.Vectors(name = args.load_emb_path,
                                  #cache = 'custom_embeddings',
                                  unk_init = torch.Tensor.normal_)
    
    
    TEXT.build_vocab(train_data,
                max_size = args.max_voc_size, 
                vectors = custom_embeddings)
    """
    TEXT.build_vocab(train_data, max_size = args.max_voc_size, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)


    test_iterator = data.BucketIterator(test_data, 
                            train = False,
                            batch_size = args.batch_size, 
                            sort_within_batch=True,
                            sort_key = lambda x: x.text,
                            device = device)
    return test_iterator, TEXT.vocab