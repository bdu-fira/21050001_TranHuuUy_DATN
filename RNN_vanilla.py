import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_vanilla(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, 
                 bidirectional, dropout, pad_idx, output_dim):
        """
        @param vocab_size (int)
        @param embedding_dim (int)
        @param hidden_dim (int)
        @param n_layers (int)
        @param bidirectional (bool)
        @param dropout (float)
        @param pad_idx (int)
        @param output_dim (int): Number of sentiment classes
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        # Sử dụng nn.RNN thay vì nn.LSTM
        self.rnn = nn.RNN(embedding_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout,
                          # non-linearity có thể là 'tanh' hoặc 'relu'
                          nonlinearity='tanh') 
        
        # Lớp fully connected sẽ nhận output từ RNN. 
        # Nếu bidirectional là True, hidden_dim * 2, ngược lại là hidden_dim
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        """
        @param text (torch.Tensor): shape = [sent len, batch size]
        @param text_lengths (torch.Tensor): shape = [batch size]
        @return
        """
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        
        # embedded = [sent len, batch size, emb dim]
        
        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        
        # Với RNN truyền thống, output chỉ có `packed_output` và `hidden`
        packed_output, hidden = self.rnn(packed_embedded)
        
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        
        # Nếu RNN là bidirectional, ta sẽ nối hidden state cuối cùng của 2 chiều
        if self.rnn.bidirectional:
            # hidden shape: [num_layers * 2, batch_size, hidden_dim]
            # Lấy hidden state của layer cuối cùng:
            # hidden[-2,:,:] là forward direction, hidden[-1,:,:] là backward direction
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            # hidden shape: [num_layers, batch_size, hidden_dim]
            hidden = self.dropout(hidden[-1,:,:])
            
        # hidden shape: [batch size, hidden_dim * num_directions]
        
        # Đưa hidden state cuối cùng qua lớp fully connected để dự đoán
        return self.fc(hidden)