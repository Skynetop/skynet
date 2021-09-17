from nltk.util import print_string
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Neuralnetwork import bag_of_words, tokenzie, stem
from Brain import NeuralNet
import nltk
from speak import say
nltk.download('punkt')
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        # print(pattern)
        w = tokenzie(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ingnore_words = [',', '?', '!', '.', '/', ]
all_words = [stem(w) for w in all_words if w not in ingnore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)
# y_train = np.all
# from listen import say

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

print("Training The Model.....!")

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()

train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
# epoch in query is reverse == str(input()) str(inpupt()) as pyautogui print("hello world")
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}')

print(f'Final loss : {loss.item():.4f}')

# this is data base of the model
data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":all_words,
    "tags":tags
}

FILE = "Training.pth"
torch.save(data,FILE)
say("training is going on arjun sir.")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
print("opertion completed!!")
say("training was completed sir!")
print(f"Training complete, File Saved to {FILE}")



