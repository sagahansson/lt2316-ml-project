import torch
import torch.nn as nn
import numpy as np
import random
import os

device = 'cuda:0'

def predict(net, word, word2ind, ind2word, h0=None, c0=None, top_k=5, device=device):
    """
    Predicts each word for write().
    net: poetrymodel.PoetryModel
    
    word2ind: dict of all vocab words (keys) with a unique int (vals)
    word: str, which word to use as primer
    ind2word: dict, opposite of word2ind
    h0: torch.Tensor, hidden state
    c0: (iff net.gru==False) torch.Tensor, cell state
    top_k: int, how many words/punctuation marks to take into consideration for each prediction
    device: str indicating which cuda to use
    
    returns: tuple consisting of the predicted word, hidden state (and iff net.gru==False, cell state)
    
    """
    
    
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
    """
    Writes a poem.
    
    net: poetrymodel.PoetryModel
    length: int, how many words/punctuation marks to generate
    word2ind: dict of all vocab words (keys)  with a unique int (vals)
    ind2word: dict, opposite of word2ind
    words: list of str, primer words
    top_k: int, how many words/punctuation marks to take into consideration for each prediction
    device: str indicating which cuda to use
    
    returns: list of str that make up a poem
    
    """
    
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
    """
    Saves poem in a txt file; if the txt file already exists, append to it.
    
    model_name: str, the name of the model
    topk: int, the top k used to generate the poem
    ws: list, return of write()
    len_ws: int, length of the prime fed to write (words).
    gen_poem: str, return of write(), but " ".join(write())
    
    """
    
    poem_save = model_name.strip(".pt") + ".txt"
    text_file = open('./models-outputs/' + poem_save, "a")

    n = text_file.write(f"\n\nModel name: {model_name}\ttopk: {topk}\twords: {ws[:len_ws]}\n")
    n = text_file.write(gen_poem)
    
    text_file.close()
    print(f"Poem saved at {poem_save}.")
    
def write_hp(hp, mod_name):
    """
    Writes hyperparameters for a model to a csv file.
    
    hp: dict containing all hyperparameters for a model.
    mod_name: str, the name of the model
    """
    if not os.path.exists('./models-outputs/' + 'hp_new.csv'):
        t = open('./models-outputs/' + 'hp_new.csv', 'w')
        w = t.write(f"Model_name, batch_size, seq_len, embedding_size, hid_size, n_epochs, lr, gru, num_layers")
        w = t.write("\n")
    else:
        t = open('./models-outputs/' + 'hp_new.csv', 'a')
    w = t.write(f"{mod_name}, {hp['batch_size']}, {hp['seq_len']}, {hp['embedding_size']}, {hp['hid_size']}, {hp['n_epochs']}, {hp['lr']}, {hp['gru']}, {hp['num_layers']}")
    w = t.write("\n")