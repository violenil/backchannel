import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from tqdm import tqdm
import os
import json
import random
"""
CNN + embeddings for backchanneling prediction.
"""
class conv_net(nn.Module):


    def __init__(self, setup, input_rows, input_cols, no_of_speakers, max, min, cuda_device, type):
        """
        Initializes the CNN.
        :param setup: JSON describing the configuration of the CNN.
        :param max: maximum value of the training dataset before scaling.
        :param min: minimum value of the training dataset before scaling.
        :param input_rows: number of mfcc features (#rows of the matrix)
        :param input_cols: number of frames (#cols of the matrix)
        """
        super().__init__()


        self.type = type

        # rounding used on the losses
        self.eps = 7

        # minimum number of epochs without loss change.
        self.check_last_losses = 5

        # set up a cuda device if available
        if torch.cuda.is_available() and cuda_device != -1:
            self.device = torch.device(f"cuda:{cuda_device}")
            print(f"Running on the GPU {cuda_device}")
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

        # Create embeddings for listeners & speakers
        embed_dim = 5
        if type == "listener" or type == "both":

            self.listener_embedding = nn.Embedding(no_of_speakers, embed_dim)
            self.list_embedding_fc = nn.Linear(embed_dim, embed_dim)
            self.l_embed_drop = nn.Dropout(p=0)  # for listener embeddings

        if type == "speaker" or type == "both":

            self.speaker_embedding = nn.Embedding(no_of_speakers, embed_dim)
            self.speak_embedding_fc = nn.Linear(embed_dim, embed_dim)
            self.s_embed_drop = nn.Dropout(p=0)  # for speaker embeddings

        # read the CNN configuraion file
        with open(setup) as f:
            config = json.load(f)
        print(config)

        convolutional_layers = []
        cnn_drop_outs = []

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

        # So that pythorch recognises the layers.
        self.convolutionals = nn.ModuleList(convolutional_layers)
        self.cnn_drop = nn.ModuleList(cnn_drop_outs)


        self.convolutionals = nn.ModuleList(convolutional_layers)
        self.cnn_drop = nn.ModuleList(cnn_drop_outs)
        # mock feature for getting the number of output features of the c.l.
        x = torch.randn(1,input_rows,input_cols).view(-1,1,input_rows, input_cols)

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

            fc_drop_outs.append(nn.Dropout(p=drop_out))
            # a -1 in the number of input neurons refers to the size of the features after the convolutional layers.
            if input == -1:
                fcs.append ( nn.Linear(self._to_linear,output) )
            else:
                fcs.append ( nn.Linear(input, output) )

        self.linears = nn.ModuleList(fcs)
        self.fc_drop = nn.ModuleList(fc_drop_outs)

        if type == "both":
            self.final_linear = nn.Linear(output+embed_dim+embed_dim, 2)

        if type == "speaker" or type == "listener":
            self.final_linear = nn.Linear(output + embed_dim, 2)


        # make this available in whatever the device is
        self.to(self.device)


    def convs(self, x):
        """
        performs the convolutions on X. Also retrieves the number of features from the output of the convolutional layers.
        :param x: input
        :return: x after the convolution.
        """

        for layer, drop in zip(self.convolutionals, self.cnn_drop):
            x = F.max_pool2d(F.relu(drop(layer(x))), (1, 2))

        if self._to_linear is None:
            print(x.shape)
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x


    def forward(self, x, list_embed, speak_embed):
        """
        feeds forward x through the CNN
        :param x: input vector
        :param embed: embedding vector (must be Long not Float)
        :return: probability of backchannel and frontchannel.
        """
        x = self.convs(x)

        x = x.view(-1, self._to_linear)

        for layer, drop in zip(self.linears[:-1], self.fc_drop[:-1]):
            x = F.relu(drop(layer(x)))

        x = self.fc_drop[-1](self.linears[-1](x))

        if self.type == "listener" or self.type == "both":
            #listener
            e1 = self.listener_embedding(list_embed)
            e1 = F.relu(self.l_embed_drop(self.list_embedding_fc(e1)))

        if self.type == "speaker" or self.type == "both":
            #speaker
            e2 = self.speaker_embedding(speak_embed)
            e2 = F.relu(self.s_embed_drop(self.speak_embedding_fc(e2)))

        if self.type == "both":
            concat = torch.cat([x,e1,e2], dim=1)
            u = self.final_linear(concat)
            return F.softmax(u, dim=1)
        elif self.type == "listener":
            concat = torch.cat([x, e1], dim=1)
            u = self.final_linear(concat)
            return F.softmax(u, dim=1)
        elif self.type == "speaker":
            concat = torch.cat([x, e2], dim=1)
            u = self.final_linear(concat)
            return F.softmax(u, dim=1)
        else:
            return F.softmax(x,dim=1)




    def fit (self, dataset, batch_size, epochs, loss_function, lr):
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
            random_idxs = list(range(0, len(dataset.X), batch_size))
            random.shuffle(random_idxs)
            for i in tqdm(random_idxs):

                # get the batches
                batch_X = dataset.X[i:i+batch_size].view(-1,1,self.input_rows, self.input_cols).to(self.device)
                batch_y = dataset.y[i:i+batch_size].to(self.device)
                batch_ls_embed = dataset.ls[i:i+batch_size].to(self.device)
                batch_sp_embed = dataset.sp[i:i + batch_size].to(self.device)

                self.zero_grad()

                # forward
                outputs = self(batch_X, batch_ls_embed, batch_sp_embed)
                

                loss = loss_function(outputs, batch_y)

                # backpropagate
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch}. Loss: {loss}")


    def predict(self, dataset, batch_size):
        """
        Calculates the accuracy of the model on a test dataset.
        :param X: test samples
        :param y: one hot encoding labels.
        :param batch_size:
        """
        self.eval()
        acc = 0
        with torch.no_grad():
            for i in tqdm(range(0,len(dataset.X),batch_size)):
                batch_X = dataset.X[i:i+batch_size].view(-1,1,self.input_rows,self.input_cols).to(self.device)
                batch_y = dataset.y[i:i+batch_size].to(self.device)

                batch_ls_embed = dataset.ls[i:i+batch_size].to(self.device) if self.type == "listener" or self.type == "both" else None
                batch_sp_embed = dataset.sp[i:i+batch_size].to(self.device) if self.type == "speaker" or self.type == "both" else None

                outputs = self(batch_X, batch_ls_embed, batch_sp_embed)
                tp = 0
                tn = 0
                fn = 0
                fp = 0
                for i,j in zip(outputs, y):
                    if torch.argmax(i) == torch.argmax(j):
                        if j.data.cpu().numpy()[0] == 1: #positive instance
                            tp += 1
                        else: 
                            tn += 1
                    else:
                        if j.data.cpu().numpy()[0] == 1:
                            fn += 1
                        else:
                            fp += 1

                #matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
                conf = [tp, fp, fn, tn]
                acc = (tp+tn)/(tp+tn+fp+fn)

        print(f"Accuracy: {round(acc,6)}")
        print(f"Confusion: TP:{int(conf[0])}, FP:{int(conf[1])}, FN:{int(conf[2])}, TN:{int(conf[3])}")



    def fwd_pass( self, X, ls_emb, sp_emb, y, optimizer, loss_function, train=False, report=True):
        """
        forwards the data, and performs backpropagation and optimiztion when `train` flag is True.
        Also reports the accuracy and loss.
        :param X: samples
        :param y: one hot encoding labels
        :param optimizer: optimized to be used.
        :param loss_function:
        :param train: flag to decide on backpropagation
        :return: accuracy, loss and confusion
        """
        if train:
            self.zero_grad()
        outputs = self(X, ls_emb, sp_emb)
        loss = loss_function(outputs, y)

        if train:
            loss.backward()
            optimizer.step()

        if report:
            tp = 0
            tn = 0
            fn = 0
            fp = 0
            for i,j in zip(outputs, y):
                if torch.argmax(i) == torch.argmax(j):
                    if j.data.cpu().numpy()[0] == 1: #positive instance
                        tp += 1
                    else: 
                        tn += 1
                else:
                    if j.data.cpu().numpy()[0] == 1:
                        fn += 1
                    else:
                        fp += 1

            #matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
            acc = (tp+tn)/(tp+tn+fp+fn)
            conf = [tp, fp, fn, tn]
            return acc, loss, conf
        else:
            return None,None


    def predict_random_chunk(self, dataset , optimizer, loss_function, size=32):
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
        random_start = numpy.random.randint(len(dataset.X)-size)
        X =  dataset.X[random_start:random_start+size].view(-1,1,self.input_rows,self.input_cols).to(self.device)
        y = dataset.y[random_start:random_start+size].to(self.device)
        ls_emb = dataset.ls[random_start:random_start+size].to(self.device) if self.type == "listener" or self.type == "both" else None
        sp_emb = dataset.sp[random_start:random_start+size].to(self.device) if self.type == "speaker" or self.type == "both" else None

        # grant no learning
        with torch.no_grad():
            acc, loss, conf = self.fwd_pass(X, ls_emb, sp_emb, y, optimizer, loss_function)
        return acc, loss, conf


    def is_learning(self, lossess, curr_loss):

        log = [ round (float(loss),self.eps) for loss in lossess [-self.check_last_losses:]]

        if log == [round(float(curr_loss),self.eps)]*self.check_last_losses:
            return False

        return True



    def reported_fit(self, train_dataset, val_dataset, loss_function, lr, batch_size, epochs,file_name, fit_tensors=False):

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
        if fit_tensors:
            train_dataset.X = train_dataset.X.to(self.device)
            train_dataset.y = train_dataset.y.to(self.device)
            if self.type == "listener" or self.type == "both":
                train_dataset.ls = train_dataset.ls.to(self.device)
            if self.type == "speaker" or self.type == "both":
                train_dataset.sp = train_dataset.sp.to(self.device)


        if not os.path.exists('reports/'):
            os.makedirs('reports/')

        optimizer = optim.Adam(self.parameters(), lr=lr)

        # track last losses to stop the training if it's not learning anything.
        losses = []

        # open file to log the accuracies.
        with open(f"reports/{file_name}","a") as f:
            for epoch in range(epochs):


                # set how the batches are gonna be forwaded.
                random_idxs = list(range(0, len(train_dataset.X), batch_size))

                random.shuffle(random_idxs)
                self.train()

                self.train()

                for i in tqdm(random_idxs):
                    # get our batches
                    batch_X = train_dataset.X[i:i+batch_size].view(-1,1,self.input_rows,self.input_cols).to(self.device)
                    batch_y = train_dataset.y[i:i+batch_size].to(self.device)
                    batch_ls_embed = train_dataset.ls[i:i + batch_size].to(self.device) if self.type == "listener" or self.type == "both" else None
                    batch_sp_embed = train_dataset.sp[i:i + batch_size].to(self.device) if self.type == "speaker" or self.type == "both" else None

                    # forward.
                    self.fwd_pass(batch_X, batch_ls_embed, batch_sp_embed, batch_y, optimizer, loss_function, train=True, report=False)

                self.eval()
                # get the number of random features to test. In the rare case where the training size is smaller than the validation
                # get the whole size of training.
                test_size = val_dataset.X.shape[0] - 1 if val_dataset.X.shape[0] - 1 < train_dataset.X.shape[0] else train_dataset.X.shape[0] - 1

                # Get the accuracies of a random chunk of the training data.
                acc, loss, conf = self.predict_random_chunk(train_dataset, optimizer, loss_function, size=test_size)


                # Get the accuracies of a random chunk of the test data.
                val_acc, val_loss, val_conf = self.predict_random_chunk(val_dataset, optimizer, loss_function, size=test_size) #conf = [tp, fp, fn, tn]

                losses.append(val_loss)

                # Print accuracies to stdout and log them into the file.
                print(
                    f"epoch: {epoch}, acc: {round(float(acc), 6)}, loss: {round(float(loss), 8)}, val_acc: {round(float(val_acc), 6)}, val_loss: {round(float(val_loss), 8)}")
                f.write(
                    f"{file_name},{epoch},{float(acc)},{float(loss)},{float(val_acc)},{float(val_loss)},{int(val_conf[0])},{int(val_conf[1])},{int(val_conf[2])},{int(val_conf[3])}\n")

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
