import ai
from config import loss_function
import dataset
import viewer


def train(args):
    """
    Train the CNN.
    :param args:
    """
    # read training samples from disk, and prepare the dataset (i.e., shuffle, scaling, and moving the data to tensors.)
    train_dataset = dataset.dataset(args['s'][0], args['s'][1], args['ss'][0], args['ss'][1])

    # same for the validation data. The data is scaled w.r.t. the values of the training data.
    val_dataset = dataset.dataset(args['v'][0], args['v'][1], args['vv'][0], args['vv'][1], train_dataset.max,train_dataset.min)

    # set up a convolutional neural network.
    net = ai.conv_net(args['C'],train_dataset.nmfcc,train_dataset.nframes,train_dataset.no_of_speakers,train_dataset.max,train_dataset.min)

    # optional flag with optional parameter.
    if args ['report'] != '':

        # fit the data and report the results.
        net.reported_fit(train_dataset, val_dataset, loss_function, args['lr'], args['b'], args['e'], args['report'])

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

    # load the test dataset from disk.
    test = dataset.dataset(args ['d'][0] , args ['d'][1], net.max, net.min)

    # predict!
    net.predict( test.X,test.y, args['b'] )

