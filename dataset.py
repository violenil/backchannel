import random
import numpy as np
import torch



"""
Holds a dataset.
"""
class dataset(object):

    def __init__(self,positive_samples_path,negative_samples_path,max = None, min = None):
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

        self.nmfcc = positive[0].shape[0]
        self.nframes = positive[0].shape[1]

        training_data = []
        for i in range(positive.shape[0]):
            training_data.append([positive[i, :, :], np.eye(2)[0]])

        for i in range(negative.shape[0]):
            training_data.append([negative[i, :, :], np.eye(2)[1]])

        # shuffle the data!
        random.shuffle(training_data)

        # move the data into tensors.
        X = torch.Tensor([i[0] for i in training_data]).view(-1, self.nmfcc, self.nframes)

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

