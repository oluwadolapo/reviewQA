import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size, n_layers = 1, dropout = 0.1):
    super(EncoderRNN, self).__init__()

    # Define class variables
    self.hidden_size = hidden_size

    # Define layers
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional = True)
    self.fc = nn.Linear(hidden_size * 2, hidden_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input_seq, input_lengths, hidden=None):
    # Convert word indexes to embeddings
    embedded = self.dropout(self.embedding(input_seq))
    # Pack padded batch of sequences for RNN module
    packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
    # Forward pass through GRU
    outputs, hidden = self.gru(packed, hidden)
    # Unpack padding
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
    # Sum bidirectional GRU outputs
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
    # Return output and final hidden state
    
    #initial decoder hidden is final hidden state of the forwards and backwards 
    #  encoder RNNs fed through a linear layer
    hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
    return outputs, hidden



class DecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, max_length=10):
    super(DecoderRNN, self).__init__()

    # Define class variables
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout

    # Define layers
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.embedding_dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input_step, last_hidden):
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    relu_out = F.relu(embedded)
    output, hidden = self.gru(relu_out, last_hidden)
    output = F.softmax(self.out(output[0]), dim=1)
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=device)



class AttnDecoderRNN1(nn.Module):
  def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, max_length=10):
    super(AttnDecoderRNN1, self).__init__()

    #Define class variables
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout

    # Define layers
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.embedding_dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input_step, last_hidden, encoder_outputs):
    # Note: we run this one step (batch of words) at a time

    # Get embedding of current input_word
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    # Forward through unidirectional GRU
    rnn_output, hidden = self.gru(embedded, last_hidden)

    # Calculate attention weights from the current GRU output

    # Element-Wise Multiply the current target decoder state with the encoder output and sum them
    attn_energies = torch.sum(rnn_output*encoder_outputs, dim=2)

    # Transpose max_length and batch_size dimensions
    attn_energies = attn_energies.t()
    # Return the softmax normalized probability score (with added dimension)
    attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)

    # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
    context = attn_weights.bmm(encoder_outputs.transpose(0,1))
    # Concatenate weighted context vector and GRU output
    rnn_output = rnn_output.squeeze(0)
    context = context.squeeze(1)
    concat_input = torch.cat((rnn_output, context), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    # Predict next word using Luong eq. 6
    output = self.out(concat_output)
    output = F.softmax(output, dim=1)
    # Return output and final hidden state
    return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        hidden = hidden.squeeze(0)
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)
        #attention = [batch size, src len]
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim = 1)

class AttnDecoderRNN2(nn.Module):
  def __init__(self, hidden_dim, output_dim, n_layers=1, dropout=0.1, max_length=10):
    super(AttnDecoderRNN2, self).__init__()

    self.attention = Attention(hidden_dim)
    self.output_dim = output_dim
    #self.embedding = nn.Embedding(output_dim, emb_dim)
    self.embedding = nn.Embedding(output_dim, hidden_dim)
    #self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
    self.rnn = nn.GRU(hidden_dim *2, hidden_dim, n_layers, dropout=(0 if n_layers == 1 else dropout))
    #self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
    self.fc_out = nn.Linear(hidden_dim*3, output_dim)
    #self.dropout = nn.Dropout(dropout
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, input_step, hidden, encoder_outputs, mask): 
    #input = [batch size]
    #hidden = [1, batch size, dec hid dim]
    #encoder_outputs = [src len, batch size, enc hid dim]
    #mask = [batch size, src len]

    # Note: we run this one step (batch of words) at a time
    # Get embedding of current input_word
    embedded = self.embedding(input_step)
    embedded = self.dropout(embedded)
    #embedded = [1, batch size, emb dim]
    a = self.attention(hidden, encoder_outputs, mask)    
    #a = [batch size, src len]
    a = a.unsqueeze(1)
    #a = [batch size, 1, src len]
    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    #encoder_outputs = [batch size, src len, enc hid dim]
    weighted = torch.bmm(a, encoder_outputs)
    #weighted = [batch size, 1, enc hid dim]
    weighted = weighted.permute(1, 0, 2)
    #weighted = [1, batch size, enc hid dim]
    rnn_input = torch.cat((embedded, weighted), dim = 2)
    #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]  
    output, hidden = self.rnn(rnn_input, hidden)
    #output = [seq len, batch size, dec hid dim * n directions]
    #hidden = [n layers * n directions, batch size, dec hid dim]
    #seq len, n layers and n directions will always be 1 in this decoder, therefore:
    #output = [1, batch size, dec hid dim]
    #hidden = [1, batch size, dec hid dim]
    #this also means that output == hidden
    assert (output == hidden).all()    
    embedded = embedded.squeeze(0)
    output = output.squeeze(0)
    weighted = weighted.squeeze(0)
    prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
    #prediction = [1, batch size, output dim]
    prediction = F.softmax(prediction, dim=1)
    return prediction, hidden