import sys

import ai
from config import loss_function
import dataset
import viewer


def get_dataset(type, positive_mfcc, negative_mfcc, positive_speakers, negative_speakers, positive_listeners,
                negative_listeners, max = None, min = None):


    if type == "both":
        return dataset.dataset(positive_mfcc, negative_mfcc, max = max, min = min, p_listeners_path = positive_listeners ,
                 p_speakers_path = positive_speakers, n_listeners_path = negative_listeners, n_speakers_path = negative_speakers )
    elif type == "speaker":
        return dataset.dataset(positive_mfcc, negative_mfcc, max = max, min = min, p_listeners_path = None ,
                 p_speakers_path = positive_speakers, n_listeners_path = None, n_speakers_path = negative_speakers )
    elif type == "listener":
        return dataset.dataset(positive_mfcc, negative_mfcc,  max = max, min = min, p_listeners_path = positive_listeners ,
                 p_speakers_path = None, n_listeners_path = negative_listeners, n_speakers_path = None )
    else:
        return dataset.dataset(positive_mfcc, negative_mfcc, max = max, min = min, p_listeners_path=None,
                               p_speakers_path=None, n_listeners_path=None, n_speakers_path=None)



def train(args):
    """
    Train the CNN.
    :param args:
    """

    # read training samples from disk, and prepare the dataset (i.e., shuffle, scaling, and moving the data to tensors.)

    train_dataset = get_dataset(args['type'],args['train_positive_mfcc'],
                                args['train_negative_mfcc'],
                                args['train_positive_speakers'],
                                args['train_negative_speakers'],
                                args['train_positive_listeners'],
                                args['train_negative_listeners'])

    # same for the validation data. The data is scaled w.r.t. the values of the training data.
    val_dataset = get_dataset(args['type'],args['val_positive_mfcc'],
                                args['val_negative_mfcc'],
                                args['val_positive_speakers'],
                                args['val_negative_speakers'],
                                args['val_positive_listeners'],
                                args['val_negative_listeners'],
                                max = train_dataset.max, min = train_dataset.min)

    # set up a convolutional neural network.

    net = ai.conv_net(args['C'],train_dataset.nmfcc,train_dataset.nframes,train_dataset.no_of_total_speakers,
                      train_dataset.max,train_dataset.min,args['cuda_device'],args['type'])


    # optional flag with optional parameter.
    if args ['report'] != '':

        try:
            # fit the data and report the results.
            net.reported_fit(train_dataset, val_dataset, loss_function, args['lr'], args['b'], args['e'], args['report'],args['fit_tensors'])

        except KeyboardInterrupt:
            pass

        # show how the data fitted w.r.t. the training and validation data.
        viewer.create_acc_loss_graph(args['report'],args['report'])

    # dont report anything.
    elif args['train'] == True:

        # just fit.
        net.fit(train_dataset.X, train_dataset.y, args['b'] , args['e'], loss_function, args['lr'])

        # spit the accuracy out.
        net.predict(val_dataset.X,val_dataset.y, args['b'])

    # save the model?
    if args ['o'] != None:

        # write it into a file.
        net.dump( args ['o'] )



def test (args):
    """
    Predict on a test dataset given a CNN model.
    :param args:
    """
    # load the CNN from disk.
    net = ai.conv_net.load(args['m'])

    if args['type'] != net.type:
        print(f"Can't dooz. Net type '{net.type}' and test type '{args['type']}' do not match.", file=sys.stderr)
        exit(-1)

    # load the test dataset from disk.

    test  = get_dataset(args['type'],
                        args['test_positive_mfcc'],
                        args['test_negative_mfcc'],
                        args['test_positive_speakers'],
                        args['test_negative_speakers'],
                        args['test_positive_listeners'],
                        args['test_negative_listeners'],
                        net.max, net.min)

    # predict!
    net.predict( test, args['b'] )

