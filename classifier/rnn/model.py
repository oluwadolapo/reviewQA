import torch
import torch.nn as nn
import torch.nn.functional as F

class lstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout, 
                    pad_idx, hidden_dim, n_layers, bidirectional):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #hidden = [batch size, hid dim * num directions]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            
        return self.fc(hidden)


class lstm_attention(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout, 
                    pad_idx, hidden_dim, n_layers, bidirectional):
        super(lstm_attention, self).__init__()
		
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                                bidirectional=bidirectional, dropout=dropout)
        self.label = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
		#self.attn_fc_layer = nn.Linear()
        
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
    
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        attn_output = self.attention_net(output, hidden)
        attn_output = self.dropout(attn_output)
        logits = self.label(attn_output)
        #import IPython; IPython.embed(); exit(1)
        return logits