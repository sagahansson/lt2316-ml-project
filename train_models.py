import pandas as pd
import torch
import torch.nn as nn
import re
import string
import random
import os
import pickle
import numpy as np
from poetrymodel import PoetryModel
device = 'cuda:0'

#class PoetryModel(nn.Module):
#    # initialize RNN module
#    def __init__(self, n_vocab, hp, dropprob=0.5):
#        super(PoetryModel, self).__init__()
#        self.seq_size = hp["seq_len"]
#        self.hid_size = hp["hid_size"]
#        self.num_layers = hp["num_layers"]
#        self.gru = hp["gru"]
#        self.embedding_size = hp["embedding_size"]
#        
#        self.embedding = nn.Embedding(n_vocab, self.embedding_size)
#        
#        if self.gru:
#            self.rnn = nn.GRU(self.embedding_size,
#                            self.hid_size,
#                            self.num_layers,
#                            dropout=dropprob,
#                            batch_first=True)
#        else:
#            self.rnn = nn.LSTM(self.embedding_size,
#                            self.hid_size,
#                            self.num_layers,
#                            dropout=dropprob,
#                            batch_first=True)
#        self.fc = nn.Linear(self.hid_size, n_vocab)
#        
#    def forward(self, x, prev_state):
#        embed = self.embedding(x)
#        output, state = self.rnn(embed, prev_state)
#        out = self.fc(output)
#
#        return out, state
#    
#    def init_zero(self, batch_size):
#        #initializing zero states for gru/lstm
#        if self.gru:
#            return (torch.zeros(self.num_layers, batch_size, self.hid_size))
#        else:
#            return (torch.zeros(self.num_layers, batch_size, self.hid_size),torch.zeros(self.num_layers, batch_size, self.hid_size))
 

def get_batches(flat_poems, w2id, batch_size, seq_len):
    
    poem_rep = [w2id[w] for w in flat_poems]
    total_batch_content = batch_size * seq_len # total num of words all batches contain
    num_batches = int(len(poem_rep) / total_batch_content)
    X = poem_rep[:num_batches*batch_size*seq_len] # input
    Y = np.zeros_like(X) # target
    Y[:-1] = X[1:]
    Y[-1] = X[0]
    
    X = np.reshape(X, (num_batches*batch_size, seq_len))
    Y = np.reshape(Y, (num_batches*batch_size, seq_len))
    
    for i in range(0, num_batches*batch_size, batch_size):
        yield X[i:i+batch_size, :], Y[i:i+batch_size, :]

def train(net, data, word2index, hp, clip=5, val_frac=0.1, print_every=1000, device='cuda:0'):
    
    
    model_name = hp["model_name"]
    epochs = hp["n_epochs"]
    batch_size = hp["batch_size"]
    seq_length = hp["seq_len"]
    lr = hp["lr"]
    
#    print(f"Epochs: {epochs}\n\
#            Batch size: {batch_size}\n\
#            Seq len: {seq_length}\n\
#            LR: {lr}")
    
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:train_idx], data[train_idx:]
    
    net = net.to(device)
    
    counter = 0 
    
    for e in range(epochs):
        if e == 0:
            print("TRAINING STARTED")
        if net.gru:
            h0 = net.init_zero(batch_size)
        else:
            h0, c0 = net.init_zero(batch_size)
            c0 = c0.to(device)
        h0 = h0.to(device)
            

        
        batches = get_batches(data, word2index, batch_size, seq_length)
        
        for x, y in batches:
            counter += 1
            
            opt.zero_grad()
            
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            
            if net.gru:
                pred, h0 = net(x, h0)
            else:
                pred, (h0, c0) = net(x, (h0, c0))
                c0 = c0.detach()
            
            h0 = h0.detach()
            loss = criterion(pred.transpose(1, 2), y)
            
            
            loss.backward(retain_graph=True)
            
            _ = nn.utils.clip_grad_norm_(net.parameters(), clip)
            
            opt.step()
            
            if counter % print_every == 0:
                if net.gru:
                    val_h0 = net.init_zero(batch_size)
                else:
                    val_h0, val_c0 = net.init_zero(batch_size)
                    val_c0 = val_c0.to(device)
                val_h0 = val_h0.to(device)
                
                
                val_losses = []
                
                net.eval()
                
                val_batches = get_batches(val_data, word2index, batch_size, seq_length)
                
                for x, y in val_batches:
                    
                    x = torch.tensor(x).to(device)
                    y = torch.tensor(y).to(device)
                    
                    if net.gru:
                        pred, val_h0 = net(x, val_h0)
                    else:
                        pred, (val_h0, val_c0) = net(x, (val_h0, val_c0))
                        val_c0 = val_c0.detach()
                    val_h0 = val_h0.detach()    
                    
                    val_loss = criterion(pred.transpose(1, 2), y)

                    
                    val_losses.append(val_loss.item())
                
                net.train()
                
                
                print(f"Epoch: {e+1}/{epochs}",
                      f"Step: {counter}",
                      f"Loss: {loss.item()}",
                      f"Val Loss: {np.mean(val_losses)}")
                text_file = open('./models-outputs/' + model_name.strip(".pt") + "_loss.txt", "a")
    
                n = text_file.write(f"\n\Epoch: {e+1}/{epochs}\tStep: {counter}\tLoss: {loss.item()}\tVal Loss: {np.mean(val_losses)}\n")
                text_file.close()
    

    
    checkpoint = {'hid_size'       : net.hid_size,
                  'n_layers'       : net.num_layers,
                  'state_dict'     : net.state_dict(),
                  'tokens'         : len(word2index),
                  'seq_size'       : net.seq_size,
                  'opt state dict' : opt.state_dict(),
                  'gru'            : net.gru,
                  'hp'             : hp
                    }
    
    
    torch.save(checkpoint, os.path.join('./models-outputs/', model_name))
    
    return model_name, loss

def row_to_hp_dict(row):
    row = list(row[1])
    hp = {
        "model_name"     : row[0],
        "batch_size"     : row[1],
        "seq_len"        : row[2],
        "embedding_size" : row[3],
        "hid_size"       : row[4],
        "n_epochs"       : row[5],
        "lr"             : row[6],
        "gru"            : row[7],
        "num_layers"     : row[8]
        }
    return hp

if __name__ == "__main__":
    
    flat_poems = pickle.load(open("flat_poems.p", "rb"))
    vocab_size = len(set(flat_poems))
    id2w = pickle.load(open("id2w.p", "rb"))
    w2id = {v : k for k, v in id2w.items()}
    
    df = pd.read_csv("./hp.csv") 
    
    for row in df.iterrows():
        #if 12 > row[0] > 9:
        hp = row_to_hp_dict(row)
        net = PoetryModel(vocab_size, hp)
        print(f"Starting training for model {hp['model_name']}.")
        model_name, loss = train(net, flat_poems, w2id, hp, print_every=1000, device=device)
        print(f"Model {hp['model_name']} done training!")
        
    