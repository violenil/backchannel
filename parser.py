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
        
        self.test_parser.add_argument('-data', metavar='datasets', type=str, nargs=1, help='''Directory where all the data lives. Format:\n
        ├── test\n
        │   ├── bc\n
        │   │   ├── data.3dmfcc.npy\n
        │   │   ├── data.mfcc.npy\n
        │   │   ├── listeners.npy\n
        │   │   └── speakers.npy\n
        │   └── fc\n
        │       ├── data.3dmfcc.npy\n
        │       ├── data.mfcc.npy\n
        │       ├── listeners.npy\n
        │       └── speakers.npy\n
        ├── train\n
        │   ├── bc\n
        │   │   ├── data.3dmfcc.npy\n
        │   │   ├── data.mfcc.npy\n
        │   │   ├── listeners.npy\n
        │   │   └── speakers.npy\n
        │   └── fc\n
        │       ├── data.3dmfcc.npy\n
        │       ├── data.mfcc.npy\n
        │       ├── listeners.npy\n
        │       └── speakers.npy\n
        └── val\n
            ├── bc\n
            │   ├── data.3dmfcc.npy\n
            │   ├── data.mfcc.npy\n
            │   ├── listeners.npy\n
            │   └── speakers.npy\n
            └── fc\n
                ├── data.3dmfcc.npy\n
                ├── data.mfcc.npy\n
                ├── listeners.npy\n
                └── speakers.npy
        ''', required=True)

        # define the training parser.

        self.train_parser = self.subparsers.add_parser('train', help='Train the CNN providing samples.')

        self.train_parser.add_argument('-C', metavar='CNN_config', type=str, help='Loads the CNN configuration.',default=CNN_CONFIG)

        self.train_parser.add_argument('-data', metavar='datasets', type=str, nargs=1, help='''Directory where all the data lives. Format:\n
        ├── test\n
        │   ├── bc\n
        │   │   ├── data.3dmfcc.npy\n
        │   │   ├── data.mfcc.npy\n
        │   │   ├── listeners.npy\n
        │   │   └── speakers.npy\n
        │   └── fc\n
        │       ├── data.3dmfcc.npy\n
        │       ├── data.mfcc.npy\n
        │       ├── listeners.npy\n
        │       └── speakers.npy\n
        ├── train\n
        │   ├── bc\n
        │   │   ├── data.3dmfcc.npy\n
        │   │   ├── data.mfcc.npy\n
        │   │   ├── listeners.npy\n
        │   │   └── speakers.npy\n
        │   └── fc\n
        │       ├── data.3dmfcc.npy\n
        │       ├── data.mfcc.npy\n
        │       ├── listeners.npy\n
        │       └── speakers.npy\n
        └── val\n
            ├── bc\n
            │   ├── data.3dmfcc.npy\n
            │   ├── data.mfcc.npy\n
            │   ├── listeners.npy\n
            │   └── speakers.npy\n
            └── fc\n
                ├── data.3dmfcc.npy\n
                ├── data.mfcc.npy\n
                ├── listeners.npy\n
                └── speakers.npy
        ''', required=True)
        
        self.train_parser.add_argument('-e', metavar='epochs', type=int, default=EPOCHS,help='Configures the #epochs')

        self.train_parser.add_argument('-lr', metavar='learning_rate', type=float, default=learning_rate,help='Sets the learning rate')

        self.train_parser.add_argument('-o', metavar='output_model', type=str,help='Provide a path to model to be saved.')

        self.train_parser.add_argument("-t","--train", action="store_true", default=True,help='Learn from the training samples and predict on the validation data')

        self.train_parser.add_argument("-r", "--report", default='', nargs='?', metavar='report_file',
        help='Learn from the training data, and iteratively predict on the validation. Get a summary of the overall process.')

        self.train_parser.set_defaults(action="train")
        self.train_parser.parse_args(['-h'])

    def parse_args(self):
        args = vars( self.parser.parse_args() )

        if 'report' in args.keys() and args ['report'] == None:
            args['report'] = REPORT_FILE

        return args
