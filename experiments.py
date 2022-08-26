from fileinput import filename
import os
import logging
from re import search

from utils.config.args import il_get_args

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import torch

from utils.BaselineModels import Algo
from utils.CreateExpert import create_expert
from utils.Domains import Domain

# Set log level for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)


'''
    Script for running all IL experiments.

    Args:
    --gpu: gpu index (default -1 - e.g., CPU)
    --data: path to the data we want to use
    --algo: Which algo we want to run [IUPE, ILPO, GAIL, BC, MOBILE]
    --exp: How many experiments to run
    --epochs: How many epoch should we run each algorithm
    --domain: Domain name (look DomainInfo.py)
    --timesteps: how many expert episodes
    --idx: idx the script should use to save the results
    --batch_size: mini batch size (even though we can fit all data in one batch, we should not do it)

    Commands examples:
    python il_experiments.py --gpu -1 --data ./Expert/expert_CartPole-v1.npz --algo IUPE --exp 1 --epochs 100 --domain cartpole --timesteps 1000 --idx 0
'''

if __name__ == '__main__':
    args = il_get_args()

    domain = Domain(args.domain)

    if args.algo == 'ALL':
        algos = Algo().get_all_algos_by_domain_and_type('IL', domain.is_continuous())
    else:
        algos = [args.algo]

    for algo in algos:
        if not os.path.exists('tmp'):
            os.makedirs('./tmp')

        if args.algo == 'IUPE':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = None

        pbar = tqdm(range(args.exp))
        for experiment in pbar:
            pbar.set_description_str(f'{algo} {experiment}', refresh=True)
            
            # e.g.: args.data => data='./Expert/expert_CartPole-v1_criteria300.npz'
            dataset_name = search("expert_(.+?).npz", args.data).group(1)
            dataset = args.data
            
            if not os.path.exists(dataset):
                print('\nExpert not found, creating one...')
                create_expert(args)

            model, env = Algo().get_algo(algo)(
                domain.get("name"),
                dataset=dataset,
                device=device,
                amount=args.timesteps,
                batch_size=args.batch_size,
                verbose=args.verbose
            )

            data = np.ndarray(shape=(0, 2))
            _pbar = tqdm(range(args.epochs))
            for epoch in _pbar:

                rewards = []
                for i in range(100):
                    done = False
                    total_reward = 0
                    env.seed(i)
                    state = env.reset()
                    while not done:
                        action, _states = model.predict(state)
                        state, reward, done, info = env.step(action)
                        total_reward += reward

                    rewards.append(total_reward)
                env.seed(None)

                data = np.append(
                    data,
                    np.array([epoch * args.timesteps, rewards])[None],
                    axis=0
                )

                if domain.is_solved() and args.early_stop:
                    if np.mean(rewards) >= domain.get('criteria'):
                        _pbar.set_postfix_str(
                            f"Mean reward {np.mean(rewards)} - Early stop",
                            refresh=True
                        )
                        break
                _pbar.set_postfix_str(f'Mean Reward: {np.mean(rewards)}', refresh=True)

                model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
            
            del model
            tf.reset_default_graph()

            fileName = f'result_{dataset_name+"_"+str(experiment)}' if args.exp > 1 and args.idx is None else f'result_{dataset_name+"_"+str(args.idx)}'
            path = f'./Results/{algo}/{args.domain}/'
            if not os.path.exists(path):
                os.makedirs(path)

            np.save(f'{path}{fileName}', data, allow_pickle=True)
