import sys

import ai
from config import loss_function
import dataset
import viewer


def get_dataset(type, args, dataset_flag, positive_flag, negative_flag, max = None, min = None):


    if type == "both":
        return dataset.dataset(args[dataset_flag][0], args[dataset_flag][1], max = max, min = min, p_listeners_path = args[positive_flag][0] ,
                 p_speakers_path = args[positive_flag][1], n_listeners_path = args[negative_flag][0], n_speakers_path = args[negative_flag][1] )
    elif type == "speaker":
        return dataset.dataset(args[dataset_flag][0], args[dataset_flag][1], max = max, min = min, p_listeners_path = None ,
                 p_speakers_path = args[positive_flag], n_listeners_path = None, n_speakers_path = args[negative_flag] )
    elif type == "listener":
        return dataset.dataset(args[dataset_flag][0], args[dataset_flag][1],  max = max, min = min, p_listeners_path = args[positive_flag] ,
                 p_speakers_path = None, n_listeners_path = args[negative_flag], n_speakers_path = None )
    else:
        return dataset.dataset(args[dataset_flag][0], args[dataset_flag][1], max = max, min = min, p_listeners_path=None,
                               p_speakers_path=None, n_listeners_path=None, n_speakers_path=None)



def train(args):
    """
    Train the CNN.
    :param args:
    """
    # read training samples from disk, and prepare the dataset (i.e., shuffle, scaling, and moving the data to tensors.)
    train_dataset = get_dataset(args['type'],args,'s','sp','sn')

    # same for the validation data. The data is scaled w.r.t. the values of the training data.
    val_dataset = get_dataset(args['type'],args,'v','vp','vn', max = train_dataset.max, min = train_dataset.min)

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

    test = get_dataset(args['type'], args, 'd', 'dp', 'dn', net.max, net.min)

    # predict!
    net.predict( test, args['b'] )

