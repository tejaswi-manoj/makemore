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
N = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
stoi['.'] = 0


for w in words:
    chs = ['.'] + list(w) + ['.'] 
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

import matplotlib.pyplot as plot 
%matplotlib inline

# Plot a 27x27 look-up table, showing how many times a combination of two letters occurs
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha = "center", va = "bottom", color = "gray")
        plt.text(j, i, N[i,j].item(), ha = "center", va = "bottom", color = "gray")
plt.axis('off')


P = N.float()
P /= P.sum(1,keepdim=True) # check broadcasting semantics to understand why this works - this operation is broadcastable
# learn broadcasting semantics well!!!!!

# Sample names from the model

g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
        print(''.join(out))

# Up until now: trained a bigram language model by counting how frequently any pairing occurs
