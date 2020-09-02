import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def tokenize(df, tokenizer):
    # Tokenize all of the sentences and map the tokens to their word IDs
    input_ids = []

    # Record the length of each sequence (after truncating to 512)
    lengths = []

    print('Tokenizing merged input sequence...')

    # For every sentence...
    for i, sen in enumerate(df.questionText):

        # Report progress
        if ((len(input_ids) % 20000) == 0):
            print('   Read {:,} comments.'.format(len(input_ids)))
        encoded_sent = tokenizer.encode(
                    sen,
                    add_special_tokens = True,
                    #max_length = 512,
                    #return_tensors = 'pt',
                    )
  
  
        for review in df['review_snippets'][i][0:5]:
            encoded_review = tokenizer.encode(
                      review,
                      add_special_tokens = False)
            for word in encoded_review:
                encoded_sent.append(word)

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

        # Record the truncated length.
        lengths.append(len(encoded_sent))
    
    print('DONE.')
    print('{:>10,} sequences'.format(len(input_ids)))
    return input_ids 


def train_data(train_inputs, validation_inputs, train_labels, 
                validation_labels, train_masks, validation_masks, bs):
    # Convert all inputs and labels into torch tensors,
    # the required datatype for our model.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create the DataLoader for training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    # Create the DataLoader for validation set
    validation_data = TensorDataset(validation_inputs, 
                        validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, 
                                sampler=validation_sampler, batch_size=bs)

    return train_dataloader, validation_dataloader

def test_data(input_ids, labels, attention_masks, bs):
    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_dataloader = DataLoader(
                  test_data,
                  sampler = SequentialSampler(test_data),
                  batch_size = bs
                  )
    return test_dataloader

def data(args, tokenizer):
    df = pd.read_json(args.data_path, orient='split')
    
    if args.local_test:
        df = df.head(12)

    input_ids = tokenize(df, tokenizer)

    # Get the labels from the DataFrame, and convert from booleans to ints.
    labels = df.is_answerable.to_numpy().astype(int)
    
    # Pad our input tokens with value 0
    input_ids = pad_sequences(input_ids, maxlen=args.max_len, dtype="long",
                          value=0, truncating="post", padding="post")
    
    # Create attention masks
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    if args.model_eval:
        test_dataloader = test_data(input_ids, labels, attention_masks, args.test_batch_size)
        return test_dataloader
    else:
        # Use 90% for training and 10% for validation
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                         random_state=2018, test_size=0.1)     
        # Do the same for the masks.
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                         random_state=2018, test_size=0.1) 
    
        train_dataloader, val_dataloader = train_data(train_inputs, validation_inputs, 
                                            train_labels, validation_labels, train_masks, 
                                                validation_masks, args.train_batch_size)
        return train_dataloader, val_dataloader
