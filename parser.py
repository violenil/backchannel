import argparse
from argparse import RawTextHelpFormatter
import os
from config import BATCH_SIZE, EPOCHS, learning_rate, CNN_CONFIG, REPORT_FILE
class parser(object):





    def __init__(self):
        """
        Defines the command-line arguments. Check the help [-h] to learn more.
        """
        self.parser = argparse.ArgumentParser(description='Backchanneling CNN.',formatter_class=RawTextHelpFormatter)
        self.parser.add_argument('-b', metavar='batch_size', type=int, default=BATCH_SIZE,help='Sets the batch size.')
        self.parser.add_argument('--cuda-device', metavar='gpu_id', type=int, default=0, help='Selects the cuda device. If -1, then CPU is selected.')

        self.subparsers = self.parser.add_subparsers(title='Mode',help='Action to perform.')

        # define the test parser.
        
        self.test_parser = self.subparsers.add_parser('test', help='Test the CNN on a dataset providing a model.',formatter_class=RawTextHelpFormatter)

        self.test_parser.add_argument('type', choices=['both', 'listener', 'speaker', 'mfcc'], type=str,
                                       help='Selects the type of network.')

        self.test_parser.set_defaults(action="test")

        self.test_parser.add_argument('-m', metavar='model', type=str,
                            help='Path to the CNN model to be loaded.',required=True)

        self.test_parser.add_argument('-test-data', metavar='dir', type=str, help='''Directory where all the data lives. Format:
<dir>
 ├── bc
 │   ├── data.mfcc.npy
 │   ├── listeners.npy
 │   └── speakers.npy
 └── fc
     ├── data.mfcc.npy
     ├── listeners.npy
     └── speakers.npy ''', required=True)



        # define the training parser.

        self.train_parser = self.subparsers.add_parser('train', help='Train the CNN providing samples.',formatter_class=RawTextHelpFormatter)

        self.train_parser.set_defaults(action="train")

        self.train_parser.add_argument('type', choices=['both','listener','speaker','mfcc'], type=str,
                                       help='Selects the type of network.')

        self.train_parser.add_argument('-C', metavar='CNN_config', type=str,
                                      help='Loads the CNN configuration.', default=CNN_CONFIG)


        self.train_parser.add_argument('-e', metavar='epochs', type=int, default=EPOCHS,
                                      help='Configures the #epochs')

        self.train_parser.add_argument('-lr', metavar='learning_rate', type=float, default=learning_rate,
                                      help='Sets the learning rate')

        self.train_parser.add_argument('-o', metavar='output_model', type=str,
                                      help='Provide a path to model to be saved.')

        self.train_parser.add_argument("-t", "--train", action="store_true", default=True,
                                      help='Learn from the training samples and predict on the validation data')

        self.train_parser.add_argument("--fit-tensors", action="store_true", default=False,
                                      help='Preloads the tensors in the GPU (if availble) to speed up the training.')

        self.train_parser.add_argument("-r", "--report", default='', nargs='?', metavar='report_file',
                                      help='Learn from the training data, and iteratively predict on the validation. Get a summary of the overall process.')


        self.train_parser.add_argument('-train-data', metavar='dir', type=str, help='''Directory where all the training data lives. Format:
<dir>
   ├── bc
   │   ├── data.mfcc.npy
   │   ├── listeners.npy
   │   └── speakers.npy
   └── fc
       ├── data.mfcc.npy
       ├── listeners.npy
       └── speakers.npy''', required=True)

        self.train_parser.add_argument('-val-data', metavar='dir', type=str, help='''Directory where all the validation data lives. Format:
<dir>
   ├── bc
   │   ├── data.mfcc.npy
   │   ├── listeners.npy
   │   └── speakers.npy
   └── fc
       ├── data.mfcc.npy
       ├── listeners.npy
       └── speakers.npy''', required=True)





    def build_args_from_dir (self, args):


        if args ['action'] == 'train':

            args['train_positive_mfcc'] = os.path.join(args['train_data'], 'bc/data.mfcc.npy')
            args['train_negative_mfcc'] = os.path.join(args['train_data'], 'fc/data.mfcc.npy')

            args['train_positive_speakers'] = os.path.join(args['train_data'], 'bc/speakers.npy')
            args['train_negative_speakers'] = os.path.join(args['train_data'], 'fc/speakers.npy')

            args['train_positive_listeners'] = os.path.join(args['train_data'], 'bc/listeners.npy')
            args['train_negative_listeners'] = os.path.join(args['train_data'], 'fc/listeners.npy')

            args['val_positive_mfcc'] = os.path.join(args['val_data'], 'bc/data.mfcc.npy')
            args['val_negative_mfcc'] = os.path.join(args['val_data'], 'fc/data.mfcc.npy')

            args['val_positive_speakers'] = os.path.join(args['val_data'], 'bc/speakers.npy')
            args['val_negative_speakers'] = os.path.join(args['val_data'], 'fc/speakers.npy')

            args['val_positive_listeners'] = os.path.join(args['val_data'], 'bc/listeners.npy')
            args['val_negative_listeners'] = os.path.join(args['val_data'], 'fc/listeners.npy')

        else:

            args['test_positive_mfcc'] = os.path.join(args['test_data'], 'bc/data.mfcc.npy')
            args['test_negative_mfcc'] = os.path.join(args['test_data'], 'fc/data.mfcc.npy')

            args['test_positive_speakers'] = os.path.join(args['test_data'], 'bc/speakers.npy')
            args['test_negative_speakers'] = os.path.join(args['test_data'], 'fc/speakers.npy')

            args['test_positive_listeners'] = os.path.join(args['test_data'], 'bc/listeners.npy')
            args['test_negative_listeners'] = os.path.join(args['test_data'], 'fc/listeners.npy')




    def parse_args(self):
        args = vars( self.parser.parse_args() )

        self.build_args_from_dir(args)

        if 'report' in args.keys() and args ['report'] == None:
            args['report'] = REPORT_FILE

        return args
