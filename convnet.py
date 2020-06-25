import os
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle

positive_samples = np.load("datasets/train/out_pos/data.mfcc.npy")
negative_samples = np.load("datasets/train/out_neg/data.mfcc.npy")
#check balanceness of the data.
print(len(positive_samples))
print(len(negative_samples))
print(positive_samples.shape)
N_FEATURES = np.concatenate(positive_samples[2]).shape[0]
training_data = []
for i in range(positive_samples.shape[0]):
        training_data.append([np.concatenate(positive_samples[i]),np.eye(2)[0]]) 

for i in range(negative_samples.shape[0]):
                training_data.append([np.concatenate(negative_samples[i]),np.eye(2)[1]])

print(len(training_data))

random.shuffle(training_data) #shuffle the data!

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = 16
        self.c2 = 32
        self.c3 = 64

        self.conv1 = nn.Conv1d(1,self.c1,5)
        self.conv2 = nn.Conv1d(self.c1,self.c2,5)
        self.conv3 = nn.Conv1d(self.c2,self.c3,5)

        x = torch.randn(N_FEATURES).view(-1,1,N_FEATURES)
        self._to_linear = None

        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,2)

    def convs(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)),2)
        x = F.max_pool1d(F.relu(self.conv2(x)),2)
        x = F.max_pool1d(F.relu(self.conv3(x)),2)

        if self._to_linear is None:
            #print(x[0].shape)
            self._to_linear = x[0].shape[0]*x[0].shape[1]

        return x

    def forward(self, x):
        x = self.convs(x)

        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

net = ConvNet().to(device)

X = torch.Tensor([i[0] for i in training_data]).view(-1, N_FEATURES)
max_training = torch.max(X)
min_training = torch.min(X)
X = (X-min_training)/(max_training-min_training) * 2-1 #scale between -1 and 1

y = torch.Tensor([i[1] for i in training_data])
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]
print(len(train_X))
print(len(test_X))

print(train_X[0])

BATCH_SIZE = 100
EPOCHS = 10
optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_function = nn.MSELoss()
def train(net):
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,N_FEATURES).to(device)
            batch_y = train_y[i:i+BATCH_SIZE].to(device)

            net.zero_grad()
            outputs = net(batch_X)
            #print(batch_y.shape)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}. Loss: {loss}")

train(net)

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1,1,N_FEATURES).to(device))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print(f"Accuracy: {round(correct/total, 3)}")

test(net)
