
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=12, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='/mnt/ChenSD/ADNet/results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--round', default='round42', type=str) 
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fps', default=False,
                        help='Whether to measure inference speed (FPS)')
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='When training was interupted, resume from the latest checkpoint')
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')

    # dataset parameters
    parser.add_argument('--batch_size', '-b', default=8, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', '-vb', default=4, type=int, help='Validation batch size')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--data_root', default='/mnt/ChenSD/dataset/') #注意修改数据路径
    parser.add_argument('--dataname', '-d', default='wind_10m_area1', type=str,
                        help='Dataset name (default: "mmnist")') #注意修改数据名称
    parser.add_argument('--pre_seq_length', default=None, type=int, help='Sequence length before prediction')
    parser.add_argument('--aft_seq_length', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--total_length', default=None, type=int, help='Total Sequence length for prediction')

    # method parameters
    parser.add_argument('--method', '-m', default='ADNet', type=str,
                        choices=[
                                 'ADNet',],
                        help='Name of video prediction method to train (default: "SimVP")') #注意修改方法
    parser.add_argument('--config_file', '-c', default='./configs/wind_10m_area1/ADNet.py', type=str,
                        help='Path to the default config file') #注意修改config_fig
    parser.add_argument('--model_type', default=None, type=str,
                        help='Name of model for SimVP (default: None)') #如果用simvp注意修改model_type

    # Training parameters
    parser.add_argument('--epoch', default=200, type=int, help='end epochs')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=200, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='LR decay rate (default: 0.1)')

    return parser
