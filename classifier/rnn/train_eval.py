import time
import math
import torch

from utils import flat_accuracy
from utils import format_time

def train(model, iterator, optimizer, criterion):
    print("")
    print('Training...')
    
    t0 = time.time()
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for step, batch in enumerate(iterator):
        optimizer.zero_grad()
        # Progress update every 1000 batches.
        if step % 1000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress
            print(' Batch {:>5,} of {:>5,}.   Elapsed: {:}.'.format(step, len(iterator), elapsed))

        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)

        # Move predictions and labels to CPU
        predictions = predictions.to(dtype=int).detach().cpu().numpy()
        labels = batch.label.to(dtype=int).detach().cpu().numpy()

        acc = flat_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # Measure how long the validation run took.
    training_time = format_time(time.time() - t0)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), training_time

def evaluate(model, iterator, criterion):
    print("")
    print('Running Validation...')

    t0 = time.time()

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            
            # Move predictions and labels to CPU
            predictions = predictions.to(dtype=int).detach().cpu().numpy()
            labels = batch.label.to(dtype=int).detach().cpu().numpy()

            acc = flat_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), validation_time

"""
def evaluate(model, iterator, criterion):
    print("")
    print('Running Evaluation...')

    t0 = time.time()

    epoch_loss = 0
    epoch_acc = 0

    # Tracking variables
    total_test_accuracy = 0
    total_test_precision = 0
    total_test_recall = 0
    total_test_f1 = 0

    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
            # Move predictionss and labels to CPU
            predictions = predictions.detach().cpu().numpy()
            labels = batch.label.to('cpu').numpy()
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), validation_time

"""