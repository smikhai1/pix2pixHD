from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--sn', action='store_true', default=False, help='Activate Spectral norm in D')

        self.parser.add_argument('--test_data_dir', type=str, default=None, help='Path to the test data')
        self.parser.add_argument('--img_size', type=int, default=256)
        self.parser.add_argument('--results_dir', type=str, default='./inference_results', help='saves results here.')
        self.parser.add_argument('--device', type=str, default='cuda')
        self.parser.add_argument('--inference_epoch_freq', type=int, default=0, help='sets the frequency of inference')
        self.parser.add_argument('--num_rows_in_grid', type=int, default=3, help='Number of rows in the results grid')

        self.parser.add_argument('--color_pres_loss_w', type=float, default=0.0,
                                 help='Weight of the color preservation loss')
        self.parser.add_argument('--color_stats_loss_w', type=float, default=0.0,
                                 help='Weight of the color statistics preservation loss')
        self.parser.add_argument('--not_save_concats', action='store_true', default=False, help='Whether not to save concats')
        self.parser.add_argument('--use_wandb', action='store_true', default=False,
                                 help='Use WandB for loggin experiment')
        self.parser.add_argument('--last_conv_zeros', action='store_true', default=False)
        self.parser.add_argument('--augs_cfg_fp', type=str, default=None,
                                 help='Path to YAML config of augs')

        self.isTrain = True
