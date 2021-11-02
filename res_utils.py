import torch
import torch.nn as nn
import numpy as np
import random
import os

device = 'cuda:0'

def predict(net, word, word2ind, ind2word, h0=None, c0=None, top_k=5, device=device):
    
    x = np.array(word2ind[word])[np.newaxis, np.newaxis]
    x = torch.LongTensor(x).to(device)
    
    if net.gru:
        out, h0 = net(x, h0)
    else:
        out, (h0, c0) = net(x, (h0, c0))
        c0 = c0.detach()
        
    h0 = h0.detach()
    
    softmax = nn.Softmax(dim=-1)
    prob = softmax(out)
    prob = prob.data.cpu()
    
    prob, top_w = prob.topk(top_k)
    
    top_w =  top_w.numpy().squeeze()
    if top_k == 1:
        top_w = [top_w]
    
    prob = (prob/prob.sum()).squeeze(0).squeeze(0)    
    word = np.random.choice(top_w, p=prob.numpy())
    
    if net.gru:
        return ind2word[word], h0
    else:
        return ind2word[word], (h0, c0)
    
def write(net, length, word2ind, ind2word, words=['the'], top_k=5, device=device):
    
    net.to(device)
    
    net.eval()
    if net.gru:
        h0 = net.init_zero(1)
    else:
        h0, c0 = net.init_zero(1)
        c0 = c0.to(device)
    h0 = h0.to(device)
    
    for word in words:
        if net.gru:
            word, h0 = predict(net, word, word2ind, ind2word, h0, top_k=top_k)
        else:
            word, (h0, c0) = predict(net, word, word2ind, ind2word, h0, c0, top_k=top_k)
        #return word
    words.append(word)
    
    for _ in range(length):
        if net.gru:
            word, h0 = predict(net, words[-1], word2ind, ind2word, h0, top_k=top_k)
        else:
            word, (h0, c0) = predict(net, words[-1], word2ind, ind2word, h0, c0, top_k=top_k)
            
        words.append(word)
    
    return words

def save_poem(model_name, topk, ws, len_ws, gen_poem):
    poem_save = model_name.strip(".pt") + ".txt"
    text_file = open('./models-outputs/' + poem_save, "a")

    n = text_file.write(f"\n\nModel name: {model_name}\ttopk: {topk}\twords: {ws[:len_ws]}\n")
    n = text_file.write(gen_poem)
    
    text_file.close()
    print(f"Poem saved at {poem_save}.")
    
def write_hp(hp, mod_name):
    if not os.path.exists('./models-outputs/' + 'hp_new.csv'):
        t = open('./models-outputs/' + 'hp.csv', 'w')
        w = t.write(f"Model_name, batch_size, seq_len, embedding_size, hid_size, n_epochs, lr, gru, num_layers")
        w = t.write("\n")
    else:
        t = open('./models-outputs/' + 'hp_new.csv', 'a')
    w = t.write(f"{mod_name}, {hp['batch_size']}, {hp['seq_len']}, {hp['embedding_size']}, {hp['hid_size']}, {hp['n_epochs']}, {hp['lr']}, {hp['gru']}, {hp['num_layers']}")
    w = t.write("\n")