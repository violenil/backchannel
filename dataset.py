import random
import sys

import numpy as np
import torch


"""
Holds a dataset.
"""
class dataset(object):

    def __init__(self,positive_samples_path,negative_samples_path,p_chan_path,n_chan_path,max = None, min = None):
        """
        Reads the backchannel (positive samples) and frontchannel (negative samples) mfcc features from disk
        and prepares them for being used in the CNN.
        :param positive_samples_path: path to the backchannels.
        :param negative_samples_path: path to the frontchannels
        :param max: used for scaling. If not provided it is calculated on the fly.
        :param min: used for scaling. If not provided it is calculated on the fly.
        """
        positive = np.load(positive_samples_path)
        negative = np.load(negative_samples_path)



        print(f"#positive samples: {len(positive)}, #negative samples: {len(negative)} ")


        #adding speakerID to data
        allSpeakers = []
        with open("../data/annotation/speaker_annotation.txt", 'r') as f:
            f.readline() # skip header
            s = f.readlines()
            for l in s:
                x = l.split('\t')[1] # keep only speakerID, since [0] is conv_chan
                if x not in allSpeakers: # only keeping unique speakerIDs
                    allSpeakers.append(x)
        allSpeakers = [x.strip('\n') for x in allSpeakers]
        speaker_to_ix = {speaker: i for i, speaker in enumerate(allSpeakers)} #This is important, creates mapping from speakerID to index

        # load lists of corresponding speakerIDs for each sample
        with open(p_chan_path, 'r') as p:
            p_speaker_id = p.readlines()
        with open(n_chan_path, 'r') as n:
            n_speaker_id = n.readlines()

        p_speaker_id = [x.strip('\n') for x in p_speaker_id]
        n_speaker_id = [x.strip('\n') for x in n_speaker_id]

        self.nmfcc = positive[0].shape[1]
        self.nframes = positive[0].shape[2]
        self.no_of_speakers = len(allSpeakers)
        self.speaker_to_ix = speaker_to_ix

        # move the data into tensors.

        # Positive samples
        Xp = torch.Tensor(positive)

        # negative samples
        Xn = torch.Tensor(negative)

        # just concatenate
        X = torch.cat((Xp,Xn),0)

        X = X.view(-1, 3, self.nmfcc, self.nframes)

        # get the permumations to shuffle X and y.
        permutations = torch.randperm(X.shape[0])

        training_data = []
        for i in range(positive.shape[0]):
            speaker_ix = speaker_to_ix[p_speaker_id[i]] #retrieves index from dictionary
            training_data.append([positive[i, :, :, :], np.eye(2)[0], speaker_ix])

        for i in range(negative.shape[0]):
            speaker_ix = speaker_to_ix[n_speaker_id[i]]
            training_data.append([negative[i, :, :, :], np.eye(2)[1], speaker_ix])
        print(training_data[0][2])
        # shuffle the data!
        X = X[permutations]

        print(X.shape)

        # create the y vector.
        y = torch.zeros((positive.shape[0]+negative.shape[0],2))
        y[:positive.shape[0], 0] = 1
        y[positive.shape[0]:, 1] = 1

        # shuffle y as well ( in the same fashion as X )
        self.y = y [ permutations ]

        print(self.y.shape)

        if max == None:
            self.max = torch.max(X)
        else:
            self.max = max

        if min == None:
            self.min = torch.min(X)
        else:
            self.min = min


        self.X = (X - self.min) / (self.max - self.min) * 2 - 1  # scale between -1 and 1

        self.y = torch.Tensor([i[1] for i in training_data])  # i[1] -> one hot encoding vector
        
        self.speaker_idx = torch.Tensor([i[2] for i in training_data]) # i[2] -> speaker indexes for samples

if __name__ == "__main__":
    pos_sample = "../test/bc/data.3dmfcc.npy"
    neg_sample = "../test/fc/data.3dmfcc.npy"
    pos_speak = "../test/bc/speakerID"
    neg_speak = "../test/fc/speakerID"
    data = dataset(pos_sample, neg_sample, pos_speak, neg_speak)
