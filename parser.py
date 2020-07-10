import argparse

from config import BATCH_SIZE, EPOCHS, learning_rate, CNN_CONFIG, REPORT_FILE
class parser(object):



    def __init__(self):
        """
        Defines the command-line arguments. Check the help [-h] to learn more.
        """
        self.parser = argparse.ArgumentParser(description='Backchanneling CNN.')
        self.parser.add_argument('-b', metavar='batch_size', type=int, default=BATCH_SIZE,help='Sets the batch size.')


        self.subparsers = self.parser.add_subparsers(title='Mode',help='Action to perform.')

        # define the test parser.
        self.test_parser = self.subparsers.add_parser('test', help='Test the CNN on a dataset providing a model.')

        self.test_parser.set_defaults(action="test")


        self.test_parser.add_argument('-m', metavar='model', type=str,
                            help='Path to the CNN model to be loaded.',required=True)

        self.test_parser.add_argument('-d', metavar=('backchannel_file','frontchannel_file'), type=str,
                                      help='Data to be tested.',nargs=2, required=True)


        # define the training parser.

        self.train_parser = self.subparsers.add_parser('train', help='Train the CNN providing samples.')

        self.train_parser.add_argument('-C', metavar='CNN_config', type=str, help='Loads the CNN configuration.',default=CNN_CONFIG)

        self.train_parser.add_argument('-s', metavar=('backchannel_file','frontchannel_file'), type=str,nargs=2,
                            help='Samples used for training the CNN. Specify both front and backchannel files.',required=True)
        self.train_parser.add_argument('-ss', metavar=('backchannel_speakerID', 'frontchannel_speakerID'), type=str, nargs=2,
                            help='Corresponding speakerID for training the CNN. Specify these for both front and backchannel files.', required=True)
        #
        self.train_parser.add_argument('-v', metavar=('backchannel_file','frontchannel_file'), type=str,nargs=2,
                                 help='Validation samples. Specify both front and backchannel files.', required=True)
        self.train_parser.add_argument('-vv', metavar=('backchannel_speakerID','fronchannel_speakerID'), type=str,nargs=2,
                                help='Corresponding speakerID for validation samples. Specify these for both fron and backchannel files.', required=True)

        self.train_parser.add_argument('-e', metavar='epochs', type=int, default=EPOCHS,help='Configures the #epochs')

        self.train_parser.add_argument('-lr', metavar='learning_rate', type=float, default=learning_rate,help='Sets the learning rate')

        self.train_parser.add_argument('-o', metavar='output_model', type=str,help='Provide a path to model to be saved.')

        self.train_parser.add_argument("-t","--train", action="store_true", default=True,help='Learn from the training samples and predict on the validation data')

        self.train_parser.add_argument("-r", "--report", default='', nargs='?', metavar='report_file',
        help='Learn from the training data, and iteratively predict on the validation. Get a summary of the overall process.')

        self.train_parser.set_defaults(action="train")

    def parse_args(self):
        args = vars( self.parser.parse_args() )

        if 'report' in args.keys() and args ['report'] == None:
            args['report'] = REPORT_FILE

        return args
