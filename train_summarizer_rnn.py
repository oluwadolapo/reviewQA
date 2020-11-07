import wandb
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import time
import json

from summarizer.rnn import config
from summarizer.rnn.data import ReadData, Vocabulary, PrepareData
from summarizer.rnn.model import EncoderRNN, DecoderRNN, AttnDecoderRNN1, AttnDecoderRNN2
from summarizer.rnn.train_eval import train, validate

def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data(args):
    Data = ReadData(args.max_length)
    pairs = Data.readFile(args)
    voc = Vocabulary("chat_data")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    
    pairs = voc.filter_voc(pairs)
    print(len(pairs))
    pre_data = PrepareData()
    batches = []
    """
    steps = len(pairs)//args.batch_size
    for i in range(steps):   
        batch = pre_data.batch2TrainData(voc, [random.choice(pairs)
          for _ in range(args.batch_size)])
        batches.append(batch)
    """
    steps = len(pairs)//args.batch_size
    for i in range(steps):
        batch = pre_data.batch2TrainData(voc, [pairs[i] for _ in range(args.batch_size)])
        batches.append(batch)

    
    # Save vocabulary for testing purpose
    vocab_json = {"voc":{}}
    vocab_json["voc"]["pad_token"] = voc.PAD_token
    vocab_json["voc"]["unk_token"] = voc.UNK_token
    vocab_json["voc"]["sos_token"] = voc.SOS_token
    vocab_json["voc"]["eos_token"] = voc.EOS_token
    vocab_json["voc"]["num_words"] = voc.num_words
    vocab_json["voc"]["word2index"] = voc.word2index
    vocab_json["voc"]["word2count"] = voc.word2count
    vocab_json["voc"]["index2word"] = voc.index2word
    with open(args.save_vocab_path, 'w') as json_data:
        json.dump(vocab_json, json_data)
        
    return voc, batches

def training(args, encoder, decoder, batches, device):
    data_split = len(batches)*90//100
    encoder_optimizer = optim.Adam(encoder.parameters(), args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), args.lr)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    best_loss = float('inf')
    for epoch in range(args.num_train_epochs):
        train_loss, encoder, decoder = train(args, batches[:data_split], encoder, decoder, encoder_optimizer, decoder_optimizer, device)
        val_loss = validate(args, batches[data_split:], encoder, decoder, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(encoder.state_dict(), args.save_encoder_path)
            torch.save(decoder.state_dict(), args.save_decoder_path)
        
        print()
        print(train_loss)
        print(val_loss)
        print()

def main():
    args = config.get_params()
    wandb.init(config=args, project=args.project_name)
    _set_random_seeds(args.random_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")

    voc, batches = prepare_data(args)

    ##### Define the encoder and decoder #####
    encoder = EncoderRNN(voc.num_words, args.hidden_size, args.encoder_n_layers, args.dropout)
    if args.with_attention:
        decoder = AttnDecoderRNN1(args.hidden_size, voc.num_words, args.decoder_n_layers, args.dropout)
    else:
        decoder = DecoderRNN(args.hidden_size, voc.num_words, args.decoder_n_layers, args.dropout)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    wandb.watch(encoder)
    wandb.watch(decoder)
    training(args, encoder, decoder, batches, device)
    print("End of Training")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')