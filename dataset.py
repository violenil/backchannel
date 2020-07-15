import random
import sys

import numpy as np
import torch



"""
Holds a dataset.
"""
class dataset(object):

    def __init__(self, positive_samples_path, negative_samples_path, speakers_path, listeners_path, max = None, min = None):
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
        speakers = np.load(speakers_path)
        listeners = np.load(listeners_path)



        print(f"#positive samples: {len(positive)}, #negative samples: {len(negative)} ")

        self.nmfcc = positive[0].shape[1]
        self.nframes = positive[0].shape[2]

        # move the data into tensors.

        # Positive samples
        Xp = torch.Tensor(positive)

        # negative samples
        Xn = torch.Tensor(negative)

        # speaker indices
        self.sp = torch.Tensor(speakers)

        # listener indices
        self.ls = torch.Tensor(listeners)

        # just concatenate
        X = torch.cat((Xp,Xn),0)

        X = X.view(-1, 3, self.nmfcc, self.nframes)

        # get the permumations to shuffle X and y.
        permutations = torch.randperm(X.shape[0])

        # shuffle the data!
        X = X[permutations]
        self.sp = self.sp [ permutations ].long()
        self.ls = self.ls [ permutations ].long()


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