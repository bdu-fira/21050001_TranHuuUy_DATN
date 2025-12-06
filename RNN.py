import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, 
                 bidirectional, dropout, pad_idx, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        rnn_dropout_arg = 0 if n_layers == 1 else dropout
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=rnn_dropout_arg)
        
        self.rnn_dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias = False)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths.to('cpu'), 
            enforce_sorted=False 
        )
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        output = self.rnn_dropout(output)
        
        # Attention
        rnn_outputs = output.permute(1, 0, 2)
        energy = torch.tanh(self.attn(rnn_outputs))
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = attention_weights.unsqueeze(1)
        context_vector = torch.bmm(attention_weights, rnn_outputs)
        context_vector = context_vector.squeeze(1)
        context_vector = self.dropout(context_vector)
        
        return self.fc(context_vector)