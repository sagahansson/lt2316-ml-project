import torch.nn as nn
import torch

class PoetryModel(nn.Module):
    
    def __init__(self, n_vocab, hp, dropprob=0.5):
        super(PoetryModel, self).__init__()
        self.seq_size = hp["seq_len"]
        self.hid_size = hp["hid_size"]
        self.num_layers = hp["num_layers"]
        self.gru = hp["gru"]
        self.embedding_size = hp["embedding_size"]
        
        self.embedding = nn.Embedding(n_vocab, self.embedding_size)
        
        if self.gru:
            self.rnn = nn.GRU(self.embedding_size,
                            self.hid_size,
                            self.num_layers,
                            dropout=dropprob,
                            batch_first=True)
        else:
            self.rnn = nn.LSTM(self.embedding_size,
                            self.hid_size,
                            self.num_layers,
                            dropout=dropprob,
                            batch_first=True)
        self.fc = nn.Linear(self.hid_size, n_vocab)
        
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.rnn(embed, prev_state)
        out = self.fc(output)

        return out, state
    
    def init_zero(self, batch_size):
        #initializing zero states for gru/lstm
        if self.gru:
            return (torch.zeros(self.num_layers, batch_size, self.hid_size))
        else:
            return (torch.zeros(self.num_layers, batch_size, self.hid_size),torch.zeros(self.num_layers, batch_size, self.hid_size))