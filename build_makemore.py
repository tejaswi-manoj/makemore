words = open('names.txt', 'r').read().splitlines() # get a list of all the names in the file

# basic information about the dataset - reading and exploring
length = len(words)
smallest_word = min(len(w) for w in words) 
longest_word = max(len(w) for w in words) 


# Bigram Language Model: one character predicts the next based on a lookup table of counts
# Simplest way: Count how often any one of the combinations occurs in the dataset. Need a dictionary for this.
# 3 things we want to check: Which are the most common starting letters? Most common ending letters? Most common letters occuring together
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>'] # first and last characters are special characters
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

# Next, we sort the entire dictionary according to the most commonly occuring bigrams

sorted(b.items(), key = lambda kv: kv[1], reverse=True) # sorted applies the kv function to each element of the list we pass 

import torch
N = torch.zeros((28,28), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

for w in words:
    chs = ['<S>'] + list(w) + ['<E>'] 
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1