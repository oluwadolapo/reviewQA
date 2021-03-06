import wandb

from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import time

from classifier.cnn import config
from classifier.cnn.train_val import train, validate
from classifier.cnn.model import CNN1d
from classifier.cnn.data import train_data
from utils import format_time
from test_classifier_cnn import test_data, evaluate

def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def optimization(model):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, criterion

def training(args, model, vocab, UNK_IDX, PAD_IDX, device, *data_iterators):
    if args.resume_training:
        model.load_state_dict(torch.load(args.load_model_path))
    else:
        #Initialize the pretrained embedding
        pretrained_embeddings = vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)

        #zero the initial weights of the unknown and padding tokens.
        EMBEDDING_DIM = args.emb_dim
        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    model = model.to(device)
    optimizer, criterion = optimization(model)
    criterion = criterion.to(device)
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    total_t0 = time.time()
    for epoch in range(1, args.n_epochs+1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.n_epochs))

        train_loss, train_acc, train_time = train(model, data_iterators[0], optimizer, criterion)
        valid_loss, valid_acc, valid_time = validate(model, data_iterators[1], criterion)

        wandb.log({"avg_train_loss": train_loss, "avg_train_acc": train_acc, "avg_val_loss": valid_loss, "avg_val_acc": valid_acc}, step=epoch)
        
        print("")
        print("   Average training loss: {0:.2f}".format(train_loss))
        print("   Training epoch took: {:}".format(train_time))

        print("   Validation Loss: {0:.2f}".format(valid_loss))
        print("   Validation took: {:}".format(valid_time))

        #if valid_loss < best_valid_loss:
        if best_valid_acc < valid_acc:
            print("Saving model to %s" % args.save_model_path)
            wandb.run.summary["best_val_loss"] = valid_loss
            wandb.run.summary["best-loss-epoch"] = epoch
            #best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), args.save_model_path)

            # Save embeddings
            print("")
            print("Saving embeddings to %s" % args.save_emb_path)
            embeddings = model.embedding.weight.data
            with open(args.save_emb_path, 'w') as f:
                for i, embedding in enumerate(tqdm(embeddings)):
                    word = vocab.itos[i]
                    #skip words with unicode symbols
                    if len(word) != len(word.encode()) or len(embedding) != args.emb_dim:
                        continue
                    vector = ' '.join([str(i) for i in embedding.tolist()])
                    f.write(f'{word} {vector}\n')
            # Test the model
            test_iterator, _ = test_data(args, device)
            evaluate(args, model, test_iterator, device)

def cnn_model(args, vocab, PAD_IDX):
    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = args.emb_dim
    N_FILTERS = args.n_filters
    FILTER_SIZES = [int(s) for s in args.filter_sizes]
    OUTPUT_DIM = 1
    DROPOUT = args.dropout
    model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
                    OUTPUT_DIM, DROPOUT, PAD_IDX)
    return model

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
    
    train_iterator, valid_iterator, vocab, UNK_IDX, PAD_IDX = train_data(args, device)

    model = cnn_model(args, vocab, PAD_IDX)
    wandb.watch(model)
    training(args, model, vocab, PAD_IDX, UNK_IDX, device, 
                train_iterator, valid_iterator)
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')