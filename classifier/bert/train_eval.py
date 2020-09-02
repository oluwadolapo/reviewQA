import math
import time
import numpy as np
import torch

from utils import format_time, flat_accuracy

def train(args, model, train_dataloader, device, optimizer, scheduler):
    print("")
    print('Training...')
    
    t0 = time.time()

    model.train()
    # Reset the total loss for this epoch.
    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        # Progress update every 1000 batches.
        if step % 1000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress
            print(' Batch {:>5,} of {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        # 'batch' contains three pytorch tensors
        # Copy each tensor to GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        loss, logits = model(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask,
                         labels=b_labels)
        
        # Accumulate loss, a tensor containing a single value
        total_train_loss += loss.item()
        
        #print('training', loss)

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Update parameters
        optimizer.step()

        # Update the learning rate
        scheduler.step()
    
    # Calculate the average loss over all of the batches
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long the validation run took.
    training_time = format_time(time.time() - t0)

    return avg_train_loss, training_time


def eval(model, val_dataloader, device):
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in val_dataloader:

        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            loss, logits = model(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask,
                         labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches
        total_eval_accuracy += flat_accuracy(pred_flat, labels_flat)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    
    # Calculate the average loss over all the batches.
    avg_val_loss = total_eval_loss / len(val_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    return avg_val_loss, avg_val_accuracy, validation_time