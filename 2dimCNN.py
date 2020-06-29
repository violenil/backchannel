import time
import parser
import ai
from config import loss_function
import dataset
import viewer

"""
Main function.
"""
def run(args):

    # read training samples from disk, and prepare the dataset (i.e., shuffle, scaling, and moving the data to tensors.)
    train_dataset = dataset.dataset(args ['p'],args ['n'])

    # same for the validation data. The data is scaled w.r.t. the values of the training data.
    val_dataset = dataset.dataset(args['vp'],args['vn'],train_dataset.max,train_dataset.min)

    # set up a convolutional neural network.
    net = ai.ConvNet(train_dataset.nmfcc,train_dataset.nframes)

    if args ['report'] == True:

        # where to save the reported accuracy, loss, etc.
        MODEL_NAME = f"model-{int(time.time())}.log"
        # fit the data and report the results.
        net.reported_fit(train_dataset.X, train_dataset.y, val_dataset.X, val_dataset.y , loss_function, args['lr'], args['b'], args['e'], MODEL_NAME)

        # show how the data fitted w.r.t. the training and validation data.
        viewer.create_acc_loss_graph(MODEL_NAME)

    # dont report anything.
    elif args['train'] == True:

        # just fit.
        net.fit(train_dataset.X, train_dataset.y, args['b'] , args['e'], loss_function, args['lr'])

        # spit the accuracy out.
        net.predict(val_dataset.X,val_dataset.y, args['b'])

if __name__ == '__main__':

    # get a command-line parser.
    p = parser.parser()

    # parse the command-line arguments.
    args = vars(p.parse_args())
    # print the arguments
    print(args)

    # call the main function
    run(args)

