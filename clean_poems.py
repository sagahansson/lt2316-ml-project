import pandas as pd
import re

def clean_poems(poem_csv, column):
    """
    Takes a csv file of poems, fetches only poem content, pads punctuation with whitespace-
    Turns strings into list of strings by splitting on spaces. 
    
    poem_csv: name of csv file containing poems
    column: name of column in poem_csv containing poems
    
    Returns: nested list of cleaned poems
    """
    df = pd.read_csv(poem_csv, index_col=0)
    poems = list(df[column])
    
    clean_poems = []
    old_eng = ["ð", "þ"] # filters out old english poems
    for poem in poems:
        contains_old_eng = any(ch in poem for ch in old_eng)
        if contains_old_eng: # poems containing either of the old_eng chars are filtered out
            continue
        poem = re.sub('(?<! )(?=[.,!?()\"])|(?<=[.,!?()\"])(?! )', r' ', poem) # pad punctuation with whitespace 
        poem = poem.split()
        clean_poem = []
        for w in poem:
            if any(c.isalpha() for c in w):
                clean_poem.append(w.lower())
            else:
                clean_poem.append(w) 
        clean_poems.append(clean_poem)
    
    return clean_poems

def clean_poems_swe(poem_txt):
    """
    Takes a txt file, pads punctuation with whitespace, removes repetitive punctuation. 
    Splits txt file on whitespace, puts all tokens in nested list (this shape is because of other prewritten functions)
    
    poem_csv: name of csv file containing poems
    column: name of column in poem_csv containing poems
    
    Returns: nested list of cleaned poems
    """
    with open(poem_txt, 'r') as f:
        poem = f.read()
    
    clean_poems = []
    
    poem = re.sub('(?<! )(?=[.,!?()\"])|(?<=[.,!?()\"])(?! )|\.(?=\.)|\-(?:[-])', r' ', poem) # pad punctuation with whitespace 
    poem = poem.split()
    
    clean_poem = []
    for i, w in enumerate(poem):
        if w == '-' and poem[i+1] == '-' and poem[i+2] == '-':
            continue
        else:
            if any(c.isalpha() for c in w): 
                clean_poem.append(w.lower())
            elif w in string.punctuation:
                clean_poem.append(w) 
    clean_poems.append(clean_poem)
    
    return clean_poems

def restructure_poems(poems):
    """
    Restructures nested list into various shapes needed for nn.
    
    poems: nested list of poetry
    
    Returns:
    flat_poems: list of poems. same as poems, except not nested.
    vocab_size: number of unique tokens in poems.
    id2w: a dict consisting of a number (key) for each unique token (value) in poems.
    w2id: opposite of id2w.
    """ 
    
    flat_poems = [w for poem in poems for w in poem]
    vocab_size = len(set(flat_poems))
    id2w = dict(enumerate(set(flat_poems)))
    w2id = {w:i for i, w in id2w.items()}
    
    return flat_poems, vocab_size, id2w, w2id


if __name__ == "__main__":
    
    poems = clean_poems("kaggle_poem_dataset.csv", "Content")
    pickle.dump(poems, open( "kaggle_poems.p", "wb" ))
    
    swe = clean_poems_swe('strindberg.txt')
    # reducing the size of the Swedish dataset to be approximately the size of the English dataset
    total = len(swe[0])
    ind = int(total/2.5)
    swe = [swe[0][:ind]]
    pickle.dump(swe, open( "swe_poems.p", "wb" ))
    
    # restructuring english data
    flat_poems, vocab_size, id2w, w2id = restructure_poems(poems)
    pickle.dump(flat_poems, open("flat_poems.p", "wb"))
    pickle.dump(id2w, open("id2w.p", "wb"))
    
    # restructuring swedihsh data
    flat_swe, swe_vocab_size, swe_id2w, swe_w2id = restructure_poems(swe)
    pickle.dump(flat_swe, open("flat_swe.p", "wb"))
    pickle.dump(swe_id2w, open("swe_id2w.p", "wb"))
    
    print("Done!")