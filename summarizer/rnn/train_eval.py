import random
import argparse

import torch

##### Define Loss function #####
def maskNLLLoss(decoder_out, target, mask):
  nTotal = mask.sum() # How many elements should we consider
  target = target.view(-1,1)
  # decoder_out shape: (batch_size, vocab_size), target_size = (batch_size, 1)
  gathered_tensor = torch.gather(decoder_out, 1, target)
  # Calculate the Negative Log Likelihood Loss
  crossEntropy = -torch.log(gathered_tensor)
  # Select the non zero elements
  loss = crossEntropy.masked_select(mask)
  # Calculate the mean of the loss
  loss = loss.mean()
  return loss, nTotal.item()


def train(args, batches, encoder, decoder, encoder_optimizer,
            decoder_optimizer, device):
  SOS_token = 2
  #Zero gradients
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  encoder.train()
  decoder.train()

  #for i in range(args.data_size//args.batch_size):
  for i in range(len(batches)):
    input_variable, lengths, target_variable, mask, max_target_len = batches[i]
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(args.batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    #use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False
    use_teacher_forcing = True
    step_loss = torch.zeros(1).to(device)
    # Forward batch of sequences through decoder one step at a time
    if use_teacher_forcing:
      for t in range(max_target_len):
        if args.with_attention:
          decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        else:
          decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # Teacher forcing: next input is current target
        decoder_input = target_variable[t].view(1, -1)
        #Calculate and accumulate loss
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        step_loss += mask_loss

      # Perform backpropagation
      step_loss.backward()
      # Adjust model weights
      encoder_optimizer.step()
      decoder_optimizer.step()
      #Zero gradients
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
  #return sum(print_losses) / n_totals
  return step_loss.item(), encoder, decoder

def validate(args, batches, encoder, decoder, device):
  encoder.eval()
  decoder.eval()
  SOS_token = 2
  # Initialize variables
  loss = []

  #for i in range(args.data_size//args.batch_size):
  for i in range(len(batches)):
    input_variable, lengths, target_variable, mask, max_target_len = batches[i]
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(args.batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    #use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False
    use_teacher_forcing = True
    step_loss = torch.zeros(1).to(device)
    # Forward batch of sequences through decoder one step at a time
    if use_teacher_forcing:
      for t in range(max_target_len):
        if args.with_attention:
          decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        else:
          decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # Teacher forcing: next input is current target
        decoder_input = target_variable[t].view(1, -1)
        #Calculate and accumulate loss
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        step_loss += mask_loss
      
      # Perform backpropagation
      #step_loss = step_loss/step_num
      loss.append(step_loss.item())
      avg_loss = sum(loss)/len(loss)
  return avg_loss
    
