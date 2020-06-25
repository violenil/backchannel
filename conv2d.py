import os
import time
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style

%matplotlib qt
style.use("ggplot")

positive_samples = np.load("../out2/out_pos2/data.mfcc.npy")
negative_samples = np.load("../out2/out_neg2/data.mfcc.npy")

#check balanceness of the data.
print(len(positive_samples))
print(len(negative_samples))
print(positive_samples.shape)

N_ROWS = positive_samples[0].shape[0]
N_COLS = positive_samples[0].shape[1]

training_data = []
for i in range(positive_samples.shape[0]):
    training_data.append([positive_samples[i,:,:],np.eye(2)[0]])
    #training_data.append([np.concatenate(positive_samples[i]),np.eye(2)[0]])

for i in range(negative_samples.shape[0]):
    training_data.append([negative_samples[i,:,:],np.eye(2)[1]])
    #training_data.append([np.concatenate(negative_samples[i]),np.eye(2)[1]])

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

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.c1, kernel_size=(13,5))

        x = torch.randn(N_ROWS, N_COLS).view(-1,1,N_ROWS,N_COLS)
        self._to_linear = None

        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv2(x)),(1,2))

        if self._to_linear is None:
            #print(x[0].shape)
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)

        x = x.view(-1, self._to_linear)# if x < 0 then 0 otherwise x.
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return F.softmax(x, dim=1)

net = ConvNet().to(device)

X = torch.Tensor([i[0] for i in training_data]).view(-1, N_ROWS, N_COLS)
max_training = torch.max(X)
min_training = torch.min(X)
X = (X-min_training)/(max_training-min_training) * 2-1 #scale between -1 and 1

y = torch.Tensor([i[1] for i in training_data]) #i[1] -> one hot encoding vector
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
#print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

#print(len(train_X))
#print(len(test_X))

#print(train_X[0])



BATCH_SIZE =1024
BATCH_TEST = BATCH_SIZE
EPOCHS 2100
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

def train(net):
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,N_ROWS, N_COLS).to(device)
            batch_y = train_y[i:i+BATCH_SIZE].to(device)

            net.zero_grad()
            outputs = net(batch_X)
            #print(batch_y.shape)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}. Loss: {loss}")

def test(net):
    correct = 0
    total = 0
    acc = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X), BATCH_TEST)):
            batch_X = test_X[i:i+BATCH_SIZE].view(-1,1,N_ROWS,N_COLS).to(device)
            batch_y = test_y[i:i+BATCH_SIZE].to(device)
            #real_class = torch.argmax(test_y[i])
            #net_out = net(test_X[i].view(-1,1,N_FEATURES).to(device))[0]
            #predicted_class = torch.argmax(net_out)
            #if predicted_class == real_class:
            #    correct += 1
            #total += 1
            outputs = net(batch_X)
            matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, batch_y)]
            acc += matches.count(True)

        acc /= len(text_X)
    print(f"Accuracy: {round(correct/total, 3)}")
def fwd_pass(X,y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss

def test(size=32):
    random_start = np.random.randint(len(test_X)-size)

    X,y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1,1,N_FEATURES).to(device), y.to(device))

    return val_acc, val_loss

train(net)
test(net)
val_acc, val_loss = test(size=100)
print(val_acc, val_loss)

MODEL_NAME = f"model-{int(time.time())}.log"

net = ConvNet().to(device)

optimizer = optim.Adam(net.parameters(),lr=0.001)
loss_function = nn.MSELoss()


print(MODEL_NAME)

def train():
    BATCH_SIZE = 512
    EPOCHS = 100
    with open(MODEL_NAME,"a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0,len(train_X),BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,N_ROWS,N_COLS).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)

                acc, loss = fwd_pass (batch_X,batch_y,train=True)
                if i % 50 == 0:
                    val_acc, val_loss = test(size=64)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")

train()

def create_acc_loss_graph():
    contents = open(MODEL_NAME,"r").read().split('\n')

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if MODEL_NAME in c:
            name,timestamp,acc,loss,val_acc,val_loss = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))
            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    fig = plt.figure()
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1),(1,0),sharex=ax1)

    ax1.plot(times,accuracies,label="acc")
    ax1.plot(times,val_accs,label="val_acc")
    ax1.legend(loc=2)

    ax2.plot(times,losses,label="loss")
    ax2.plot(times,val_losses,label="val_loss")
    ax2.legend(loc=2)

    plt.show()
create_acc_loss_graph()
