import argparse
import os
import warnings

from utils.BaselineModels import Algo
from utils.Domains import get_all_domains


number_of_experiments = 10
amount_of_timesteps = 1000
amount_of_epochs = 1000  # total timesteps = timesteps * epochs


def rl_get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Args for general configs')

    parser.add_argument(
        '--gpu',
        default='-1',
        help='GPU Number.'
    )
    parser.add_argument(
        '--algo',
        choices=Algo().get_all_algos_by_type('RL'),
        type=str,
        help='RL algorithm use in the experiment'
    )
    parser.add_argument(
        '--domain',
        choices=get_all_domains(),
        type=str,
        help='Environment to be use in the experiment'
    )
    parser.add_argument(
        '--exp',
        default=number_of_experiments,
        type=int,
        help=''
    )
    parser.add_argument(
        '--epochs',
        default=amount_of_epochs,
        type=int,
        help='Amount of measurements done'
    )
    parser.add_argument(
        '--timesteps',
        default=amount_of_timesteps,
        type=int,
        help='Amount of samples between measurements'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether or not to print during training'
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    return args


def il_get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Args for general configs')

    parser.add_argument(
        '--gpu',
        default='-1',
        help='GPU Number.'
    )
    parser.add_argument(
        '--data',
        help='Path to the expert data'
    )
    parser.add_argument(
        '--algo',
        choices=Algo().get_all_algos_by_type('IL'),
        type=str,
        help='IL algorithm use in the experiment'
    )
    parser.add_argument(
        '--domain',
        choices=get_all_domains(),
        type=str,
        help='Environment to be use in the experiment'
    )
    parser.add_argument(
        '--exp',
        default=number_of_experiments,
        type=int,
        help=''
    )
    parser.add_argument(
        '--epochs',
        default=amount_of_epochs,
        type=int,
        help='Amount of measurements done'
    )
    parser.add_argument(
        '--timesteps',
        default=amount_of_timesteps,
        type=int,
        help='Amount of samples between measurements'
    )
    parser.add_argument(
        '--early_stop',
        action='store_true',
        help='Whether or not to print during training'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether or not to print during training'
    )
    parser.add_argument(
        '--path',
        default=None
    )
    parser.add_argument(
        '--batch_size',
        default=500,
        type=int
    )
    parser.add_argument(
        '--random',
        default=None
    )
    parser.add_argument(
        '--idx',
        default=None,
        type=int
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    return args