import numpy as np
import random 
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import stem, tokenize, bag_word
from Nino_model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for p in intent['patterns']:
        w = tokenize(p)
        words.extend(w)
        xy.append((w, tag)) #input x will be w and output y is tag

ignore_w = ['!', '?', '.', ',', '>', '<', ')', '(', '[', ']', '{','}', '|']
#stem and lower words
words = [stem(w) for w in words if w not in ignore_w]
#remove duplicate
words = sorted(set(words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (pattern, tag) in xy:
    train_data = bag_word(pattern, words)
    X_train.append(train_data) #covert a sentence "Hi there" to array [1,1]
    #y must to be a label like 0 and 1
    label = tags.index(tag)
    Y_train.append(label)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
#setting parameters
epochs = 1000
batch_size = 8
learn_rate = 0.001
inp_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
class chatData(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples
    
data = chatData()
train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inp_size, hidden_size, output_size).to(device)

#optimize
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), learn_rate)

#Train model
for ep in range(epochs):
    for (word, label) in train_loader:
        word = word.to(device)
        label = label.to(device, dtype=torch.int64)

        #Forward pass
        outputs = model(word)
        loss = crit(outputs, label)

        #optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
    #print loss per 100 epochs
    if (ep+1) % 100 == 0:
        print(f'Epoch [{ep+1}/{epochs}], Loss: {loss.item():.5f}')
print(f'Final loss: {loss.item():.5f}')

data={
    "model_state": model.state_dict(),
    "input_size": inp_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "words": words,
    "tags": tags
}

file="Nino_data.pth"
torch.save(data, file)
print(f'Training done, file: {file}')