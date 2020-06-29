import argparse

from config import BATCH_SIZE, EPOCHS, learning_rate
class parser(object):



    def __init__(self):
        """
        Defines the command-line arguments. Check the help [-h] to learn more.
        """
        self.parser = argparse.ArgumentParser(description='Backchanneling CNN.')
        self.parser.add_argument('-p', metavar='backchannel_samples', type=str,
                            help='Backchannel samples used for training the CNN',required=True)
        self.parser.add_argument('-n', metavar='frontchannel_samples', type=str,
                            help='Frontchannel samples used for training the CNN',required=True)
        #
        self.parser.add_argument('-vp', metavar='backchannel_validation_samples', type=str,
                                 help='Backchannel samples for testing the model', required=True)

        self.parser.add_argument('-vn', metavar='frontchannel_validation_samples', type=str,
                                 help='Frontchannel samples for testing the model', required=True)

        self.parser.add_argument('-b', metavar='batch_size', type=int, default=BATCH_SIZE,help='Sets the batch size')
        self.parser.add_argument('-e', metavar='epochs', type=int, default=EPOCHS,help='Configures the #epochs')
        self.parser.add_argument('-lr', metavar='learning_rate', type=float, default=learning_rate,help='Sets the learning rate')

        self.parser.add_argument("-t","--train", action="store_true", default=True,help='Learn from the training samples and predict on the validation data')
        self.parser.add_argument("-r", "--report", action="store_true", default=False,help='Learn from the training data, and iteratively predict on the validation. Get a summary of the overall process.')
        #parser.print_help()

    def parse_args(self):
        return self.parser.parse_args()