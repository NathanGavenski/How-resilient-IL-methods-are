import argparse
import sys
import os

import gdown


def download_file(url, output, env_name, quiet=False):

    if not os.path.isdir(output):
        os.makedirs(output)
        print("created folder: ", output)

    for i, (name, link) in enumerate(url.items()):
        print(name, link)
        print(f'\nDownloading file {i+1}/{len(url)}.')
        link = 'https://drive.google.com/uc?id=' + link
        gdown.download(link, output + name , quiet)


def define_expert(env_name):
    if env_name.lower() == 'all':
        return {
            'expert_Acrobot-v1.npz': '1qP2iiX1iHhQubNOHWoeEgelUBrya0INd',
            'expert_CartPole-v1.npz': '1qNYUjTIYf779zhWei43DEfsdFfwonGCD',
            'expert_LunarLander-v2.npz': '1h_kB_EugnnHGvKvGlYnEhwrhosFKwK9I',
            'expert_MountainCar-v0.npz': '1stLWWILngIwm6GX4mye-8E03SE7_aNOS',
        }

    elif env_name.lower() == 'acrobot':
        return { 'expert_Acrobot-v1.npz': '1qP2iiX1iHhQubNOHWoeEgelUBrya0INd' }

    elif env_name.lower() == 'cartpole':
        return { 'expert_CartPole-v1.npz': '1qNYUjTIYf779zhWei43DEfsdFfwonGCD' }

    elif env_name.lower() == 'lunarlander':
        return { 'expert_LunarLander-v2.npz': '1h_kB_EugnnHGvKvGlYnEhwrhosFKwK9I' }

    elif env_name.lower() == 'mountaincar':
        return { 'expert_MountainCar-v0.npz': '1stLWWILngIwm6GX4mye-8E03SE7_aNOS' }

    elif env_name.lower() == 'experiments':
        return { 'ablation.zip': '1xxoenA_i9kraBtgfwX_RfPY8pOkYtBZd' }

    else:
        print('Please insert a valid argument for env_name')
        sys.exit()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", default='all', help="environment name", choices=[
                        "all", "acrobot", "cartpole", "lunarlander", "mountaincar", "experiments"], type=str)

    parser.add_argument("--output", default='./Expert/',
                        help="path to store downloaded experts", type=str)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    env = args.env_name
    url = define_expert(env)
    destination = args.output
    download_file(url, destination, env)


if __name__ == '__main__':
    main()
