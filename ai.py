import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import json


"""
CNN for backchanneling.
"""
class conv_net(nn.Module):

    def __init__(self, setup, input_rows, input_cols, max, min):
        """
        Initializes the CNN.
        :param setup: JSON describing the configuration of the CNN.
        :param max: maximum value of the training dataset before scaling.
        :param min: minimum value of the training dataset before scaling.
        :param input_rows: number of mfcc features (#rows of the matrix)
        :param input_cols: number of frames (#cols of the matrix)
        """
        super().__init__()

        # set up a cuda device if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")

        # values used to scale the training dataset.
        self.max = max
        self.min = min

        # i.e. number of mfcc
        self.input_rows = input_rows
        # number of frames
        self.input_cols = input_cols

        # read the CNN configuraion file
        with open(setup) as f:
            config = json.load(f)
        print(config)
        print(type(config))

        convolutional_layers = []

        # set up convolutional layer(s).
        for cname, properties in config['convolutions'].items():
            in_channels = properties['in_channels']
            out_channels = properties['out_channels']
            kernel_width = properties['kernel_width']
            convolutional_layers.append( nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(input_rows,kernel_width)) )

        # So that pythorch recognises the layers.
        self.convolutionals = nn.ModuleList(convolutional_layers)

        # mock feature for getting the number of output features of the c.l.
        x = torch.randn(input_rows,input_cols).view(-1,1,input_rows, input_cols)


        self._to_linear = None

        # Get the output features of the c.l.
        self.convs(x)

        # fully connected layers
        fcs = []

        # set up the linear layer(s).
        for cname, properties in config['fully_connected'].items():
            input = properties['input']
            output = properties['output']

            # a -1 in the number of input neurons refers to the size of the features after the convolutional layers.
            if input == -1:
                fcs.append ( nn.Linear(self._to_linear,output) )
            else:
                fcs.append ( nn.Linear(input, output) )

        self.linears = nn.ModuleList(fcs)

        # make this available in whatever the device is
        self.to(self.device)


    def convs(self, x):
        """
        performs the convolutions on X. Also retrieves the number of features from the output of the convolutional layers.
        :param x: input
        :return: x after the convolution.
        """
        for layer in self.convolutionals:
            x = F.max_pool2d(F.relu(layer(x)),(1,2))

        if self._to_linear is None:
            print(x[0].shape)
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x


    def forward(self, x):
        """
        feeds forward x through the CNN
        :param x: input vector
        :return: probability of backchannel and frontchannel.
        """
        x = self.convs(x)

        x = x.view(-1, self._to_linear)

        for layer in self.linears[:-1] :
            x = F.relu(layer(x))

        x = self.linears[-1] (x)

        return F.softmax(x, dim=1)



    def fit (self, X, y, batch_size, epochs, loss_function, lr):
        """
        Trains the CNN.
        :param X: training samples
        :param y: one hot encoding labels.
        :param batch_size:
        :param ephocs:
        :param loss_function:
        :param lr: learning rate.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            for i in tqdm(range(0, len(X), batch_size)):

                # get the batches
                batch_X = X[i:i+batch_size].view(-1,1,self.input_rows, self.input_cols).to(self.device)
                batch_y = y[i:i+batch_size].to(self.device)

                self.zero_grad()

                # forward
                outputs = self(batch_X)

                loss = loss_function(outputs, batch_y)

                # backpropagate
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch}. Loss: {loss}")


    def predict(self, X, y, batch_size):
        """
        Calculates the accuracy of the model on a test dataset.
        :param X: test samples
        :param y: one hot encoding labels.
        :param batch_size:
        """
        correct = 0
        total = 0
        acc = 0
        with torch.no_grad():
            for i in tqdm(range(0,len(X),batch_size)):
                batch_X = X[i:i+batch_size].view(-1,1,self.input_rows,self.input_cols).to(self.device)
                batch_y = y[i:i+batch_size].to(self.device)
                #real_class = torch.argmax(test_y[i])
                #net_out = net(test_X[i].view(-1,1,N_FEATURES).to(device))[0]
                #predicted_class = torch.argmax(net_out)
                #loss = loss_function(outputs,y)
                outputs = self(batch_X)

                matches = [ torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs,batch_y)]
                acc += matches.count(True)

                #if predicted_class == real_class:
                    #correct += 1
                #total += 1
            acc /= len(X)

        print(f"Accuracy: {round(acc,3)}")



    def fwd_pass( self, X, y, optimizer, loss_function, train=False):
        """
        forwards the data, and performs backpropagation and optimiztion when `train` flag is True.
        Also reports the accuracy and loss.
        :param X: samples
        :param y: one hot encoding labels
        :param optimizer: optimized to be used.
        :param loss_function:
        :param train: flag to decide on backpropagation
        :return: accuracy and loss.
        """
        if train:
            self.zero_grad()
        outputs = self(X)
        matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
        acc = matches.count(True)/len(matches)
        loss = loss_function(outputs, y)

        if train:
            loss.backward()
            optimizer.step()

        return acc, loss


    def test_chunk(self, X, y , optimizer, loss_function, size=32):
        """
        Get the accuracy and lost a random chunk of test data.
        :param X: test samples
        :param y: one hot encoded vector.
        :param optimizer:
        :param loss_function:
        :param size: size of the random chunk.
        :return: accuracy and loss.
        """
        # get a random chunk from the test data
        random_start = np.random.randint(len(X)-size)
        X,y = X[random_start:random_start+size], y [random_start:random_start+size]

        # grant no learning
        with torch.no_grad():
            acc, loss = self.fwd_pass(X.view(-1,1,self.input_rows,self.input_cols).to(self.device),y.to(self.device), optimizer, loss_function)
        return acc, loss

    def reported_fit(self, X_train, y_train, X_val, y_val, loss_function, lr, batch_size, epochs,file_name):
        """
        Trains the CNN and reports accuracy and loss on both validation and training data at each epoch. The reported data is also
        saved in a csv file.
        :param X_train: training samples.
        :param y_train: one hot encoded labels.
        :param X_val: validation samples.
        :param y_val: one hot encoded labels
        :param loss_function:
        :param lr: learning rate.
        :param batch_size:
        :param epochs:
        :param file_name: write the reported accuracies and losses into this file.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        with open(file_name,"a") as f:
            for epoch in range(epochs):
                for i in tqdm(range(0,len(X_train),batch_size)):
                    batch_X = X_train[i:i+batch_size].view(-1,1,self.input_rows,self.input_cols).to(self.device)
                    batch_y = y_train[i:i+batch_size].to(self.device)

                    acc, loss = self.fwd_pass (batch_X,batch_y,optimizer,loss_function, train=True)

                val_acc, val_loss = self.test_chunk(X_val, y_val, optimizer, loss_function, size=X_val.shape[0] - 1)
                print(f"acc: {round(float(acc), 2)}, loss: {round(float(loss), 4)}, val_acc: {round(float(val_acc), 2)}, val_loss: {round(float(val_loss), 4)}")
                f.write(
                    f"{file_name},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)},{round(float(val_acc), 2)},{round(float(val_loss), 4)}\n")

    def dump(self, path):
        """
        Write the model into a file.
        :param path: path to the file to be written.
        """
        torch.save(self,path)

    @staticmethod
    def load(path):
        """
        Loads a model from a file.
        :param path:
        :return: CNN with weights already set up.
        """
        return torch.load(path)