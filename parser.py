import argparse

from config import BATCH_SIZE, EPOCHS, learning_rate, CNN_CONFIG, REPORT_FILE
class parser(object):





    def __init__(self):
        """
        Defines the command-line arguments. Check the help [-h] to learn more.
        """
        self.parser = argparse.ArgumentParser(description='Backchanneling CNN.')
        self.parser.add_argument('-b', metavar='batch_size', type=int, default=BATCH_SIZE,help='Sets the batch size.')
        self.parser.add_argument('--cuda-device', metavar='gpu_id', type=int, default=0, help='Selects the cuda device. If -1, then CPU is selected.')

        self.subparsers = self.parser.add_subparsers(title='Mode',help='Action to perform.')

        # define the test parser.
        self.test_parser = self.subparsers.add_parser('test', help='Test the CNN on a dataset providing a model.')

        self.test_parser.set_defaults(action="test")

        self.test_parser.add_argument('-m', metavar='model', type=str,
                            help='Path to the CNN model to be loaded.',required=True)

        self.test_parser.add_argument('-d', metavar=('backchannel_file','frontchannel_file'), type=str,
                                      help='Data to be tested.',nargs=2, required=True)


        self.subtest_parser = self.test_parser.add_subparsers(title='Type', help='Select the features to use for testing.')


        # both embeddings parser
        self.test_both_embeddings_parser = self.subtest_parser.add_parser('both', help='Test the network using mfcc, speaker and listener embeddings.')

        self.test_both_embeddings_parser.set_defaults(type="both")

        self.test_both_embeddings_parser.add_argument('-dp', metavar=('listener_indices', 'speaker_indices'), type=str, nargs=2,
                            help='Corresponding speaker/listener indices for positive samples.', required=True)
        self.test_both_embeddings_parser.add_argument('-dn', metavar=('listener_indices', 'speaker_indices'), type=str, nargs=2,
                            help='Corresponding speaker/listener indices for negative samples.', required=True)


        # listener embeddings parser

        self.test_listener_parser = self.subtest_parser.add_parser('listener',
                                                                      help='Test the network using mfcc and listener embeddings.')

        self.test_listener_parser.set_defaults(type="listener")

        self.test_listener_parser.add_argument('-dp', metavar='listener_indices', type=str,
                                                 help='Corresponding listener indices for positive samples.',
                                                 required=True)
        self.test_listener_parser.add_argument('-dn', metavar='listener_indices', type=str,
                                                 help='Corresponding listener indices for negative samples.',
                                                 required=True)

        # speaker embeddings parser

        self.test_speaker_parser = self.subtest_parser.add_parser('speaker',
                                                               help='Test the network using mfcc and speaker embeddings.')

        self.test_speaker_parser.set_defaults(type="speaker")

        self.test_speaker_parser.add_argument('-dp', metavar='speaker_indices', type=str,
                                                 help='Corresponding speaker indices for positive samples.',
                                                 required=True)
        self.test_speaker_parser.add_argument('-dn', metavar='speaker_indices', type=str,
                                                 help='Corresponding speaker indices for negative samples.',
                                                 required=True)

        # mfcc parser

        self.test_mfcc_parser = self.subtest_parser.add_parser('mfcc',
                                                               help='Test the CNN using just the mfcc features.')

        self.test_mfcc_parser.set_defaults(type="mfcc")






        # define the training parser.

        self.train_parser = self.subparsers.add_parser('train', help='Train the CNN providing samples.')

        self.train_parser.set_defaults(action="train")

        self.train_parser.add_argument('-C', metavar='CNN_config', type=str,
                                      help='Loads the CNN configuration.', default=CNN_CONFIG)

        self.train_parser.add_argument('-s', metavar=('backchannel_file', 'frontchannel_file'), type=str,
                                      nargs=2,
                                      help='Samples used for training the CNN. Specify both front and backchannel files.',
                                      required=True)
        #
        self.train_parser.add_argument('-v', metavar=('backchannel_file', 'frontchannel_file'), type=str,
                                      nargs=2,
                                      help='Validation samples. Specify both front and backchannel indices path.',
                                      required=True)

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



        self.subtrain_parser = self.train_parser.add_subparsers(title='Type', help='Select the features to use.')


        # both embeddings parser
        self.both_embeddings_parser = self.subtrain_parser.add_parser('both', help='Train the network using mfcc, speaker and listener embeddings.')

        self.both_embeddings_parser.set_defaults(type="both")

        self.both_embeddings_parser.add_argument('-sp', metavar=('listener_indices', 'speaker_indices'), type=str, nargs=2,
                            help='Corresponding speaker/listener indices for positive samples.', required=True)
        self.both_embeddings_parser.add_argument('-sn', metavar=('listener_indices', 'speaker_indices'), type=str, nargs=2,
                            help='Corresponding speaker/listener indices for negative samples.', required=True)
        #
        self.both_embeddings_parser.add_argument('-vp', metavar=('listener_indices','speaker_indices'), type=str,nargs=2,
                                help='Corresponding speaker/listener indices for positive validation samples.', required=True)
        self.both_embeddings_parser.add_argument('-vn', metavar=('listener_indices','speaker_indices'), type=str,nargs=2,
                                help='Corresponding speaker/listener indices for negative validation samples.', required=True)



        # listener embeddings parser

        self.listener_parser = self.subtrain_parser.add_parser('listener',
                                                                      help='Train the network using mfcc and listener embeddings.')

        self.listener_parser.set_defaults(type="listener")

        self.listener_parser.add_argument('-sp', metavar='listener_indices', type=str,
                                                 help='Corresponding listener indices for positive samples.',
                                                 required=True)
        self.listener_parser.add_argument('-sn', metavar='listener_indices', type=str,
                                                 help='Corresponding listener indices for negative samples.',
                                                 required=True)

        self.listener_parser.add_argument('-vp', metavar='listener_indices', type=str,
                                                 help='Corresponding listener indices for positive validation samples.',
                                                 required=True)
        self.listener_parser.add_argument('-vn', metavar='listener_indices', type=str,
                                                 help='Corresponding listener indices for negative validation samples.',
                                                 required=True)


        # speaker embeddings parser

        self.speaker_parser = self.subtrain_parser.add_parser('speaker',
                                                               help='Train the network using mfcc and speaker embeddings.')

        self.speaker_parser.set_defaults(type="speaker")

        self.speaker_parser.add_argument('-sp', metavar='speaker_indices', type=str,
                                                 help='Corresponding speaker indices for positive samples.',
                                                 required=True)
        self.speaker_parser.add_argument('-sn', metavar='speaker_indices', type=str,
                                                 help='Corresponding speaker indices for negative samples.',
                                                 required=True)
        self.speaker_parser.add_argument('-vp', metavar='speaker_indices', type=str,
                                                 help='Corresponding speaker indices for positive validation samples.',
                                                 required=True)
        self.speaker_parser.add_argument('-vn', metavar='speaker_indices', type=str,
                                                 help='Corresponding speaker indices for negative validation samples.',
                                                 required=True)

        # mfcc parser

        self.mfcc_parser = self.subtrain_parser.add_parser('mfcc',
                                                               help='Train the CNN using just the mfcc features.')

        self.mfcc_parser.set_defaults(type="mfcc")







    def parse_args(self):
        args = vars( self.parser.parse_args() )

        if 'report' in args.keys() and args ['report'] == None:
            args['report'] = REPORT_FILE

        return args
