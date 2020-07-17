import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import json
import random


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

        # rounding used on the losses
        self.eps = 7

        # minimum number of epochs without loss change.
        self.check_last_losses = 5

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

        cnn_drop_outs = []

        # set up convolutional layer(s).
        for cname, properties in config['convolutions'].items():
            in_channels = properties['in_channels']
            out_channels = properties['out_channels']
            kernel_width = properties['kernel_width']
            if 'drop_out' in properties:
                drop_out = float(properties['drop_out'])
            else:
                drop_out = 0.0

            cnn_drop_outs.append(nn.Dropout(p=drop_out))
            convolutional_layers.append( nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(input_rows,kernel_width)) )

        # So that pytorch recognises the layers and drop_outs.

        self.convolutionals = nn.ModuleList(convolutional_layers)
        self.cnn_drop = nn.ModuleList(cnn_drop_outs)
        # mock feature for getting the number of output features of the c.l.
        x = torch.randn(input_rows,input_cols).view(-1,1,input_rows, input_cols)


        self._to_linear = None

        # Get the output features of the c.l.
        self.convs(x)

        # fully connected layers
        fcs = []
        fc_drop_outs = []
        # set up the linear layer(s).
        for cname, properties in config['fully_connected'].items():
            input = properties['input']
            output = properties['output']
            if 'drop_out' in properties:
                drop_out = float(properties['drop_out'])
            else:
                drop_out = 0.0


            fc_drop_outs.append(nn.Dropout(p=drop_out))
            # a -1 in the number of input neurons refers to the size of the features after the convolutional layers.
            if input == -1:
                fcs.append ( nn.Linear(self._to_linear,output) )
            else:
                fcs.append ( nn.Linear(input, output) )

        self.linears = nn.ModuleList(fcs)
        self.fc_drop = nn.ModuleList(fc_drop_outs)


        # make this available in whatever the device is
        self.to(self.device)


    def convs(self, x):
        """
        performs the convolutions on X. Also retrieves the number of features from the output of the convolutional layers.
        :param x: input
        :return: x after the convolution.
        """
        for layer, drop in zip(self.convolutionals,self.cnn_drop):
            x = F.max_pool2d(F.relu( drop( layer(x) ) ) ,(1,2))

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

        for layer,drop in zip(self.linears[:-1], self.fc_drop[:-1] ):
            x = F.relu( drop( layer(x) ) )

        x = self.fc_drop [-1] ( self.linears[-1] (x) )

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
        self.train()

        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):

            # set how the batches are gonna be forwaded.
            random_idxs = list(range(0, len(X), batch_size))

            random.shuffle(random_idxs)
            for i in tqdm(random_idxs):

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
        self.eval()
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



    def fwd_pass( self, X, y, optimizer, loss_function, train=False, report=True):
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
        loss = loss_function(outputs, y)

        if train:
            loss.backward()
            optimizer.step()

        if report:
            matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
            acc = matches.count(True)/len(matches)
            return acc, loss
        else:
            return None,None


    def predict_random_chunk(self, X, y , optimizer, loss_function, size=32):
        """
        Get the accuracy and lost a random chunk of data.
        :param X: samples to be predicted
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


    def is_learning(self, lossess, curr_loss):

        log = [ round (float(loss),self.eps) for loss in lossess [-self.check_last_losses:]]

        if log == [round(float(curr_loss),self.eps)]*self.check_last_losses:
            return False

        return True


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

        if not os.path.exists('reports/'):
            os.makedirs('reports/')

        optimizer = optim.Adam(self.parameters(), lr=lr)

        # track last losses to stop the training if it's not learning anything.
        losses = []

        # open file to log the accuracies.
        with open(f"reports/{file_name}","a") as f:
            for epoch in range(epochs):

                # set how the batches are gonna be forwaded.
                random_idxs = list(range(0, len(X_train), batch_size))

                random.shuffle(random_idxs)
                self.train()
                for i in tqdm(random_idxs):

                    # get our batches
                    batch_X = X_train[i:i+batch_size].view(-1,1,self.input_rows,self.input_cols).to(self.device)
                    batch_y = y_train[i:i+batch_size].to(self.device)

                    # forward.
                    self.fwd_pass(batch_X, batch_y, optimizer, loss_function, train=True, report=False)

                self.eval()
                # get the number of random features to test. In the rare case where the training size is smaller than the validation
                # get the whole size of training.
                test_size = X_val.shape[0] - 1 if X_val.shape[0] - 1 < X_train.shape[0] else X_train.shape[0] - 1

                # Get the accuracies of a random chunk of the training data.
                acc, loss = self.predict_random_chunk(X_train, y_train, optimizer, loss_function, size=test_size)


                # Get the accuracies of a random chunk of the test data.
                val_acc, val_loss = self.predict_random_chunk(X_val, y_val, optimizer, loss_function, size=test_size)

                losses.append(val_loss)

                # Print accuracies to stdout and log them into the file.
                print(f"epoch: {epoch}, acc: {round(float(acc), 6)}, loss: {round(float(loss), 8)}, val_acc: {round(float(val_acc), 6)}, val_loss: {round(float(val_loss), 8)}")
                f.write(
                    f"{file_name},{epoch},{float(acc)},{float(loss)},{float(val_acc)},{float(val_loss)}\n")

                if not self.is_learning(losses,val_loss):
                    print("I'm not learning :(. Try other hyperparameters. Stopping.")
                    return

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