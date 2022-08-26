import argparse
import os
import warnings
from copy import deepcopy
import sys

import gym
import numpy as np
import stable_baselines
from tqdm import tqdm

from utils.Domains import Domain, get_all_domains


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--domain',
        choices=get_all_domains(),
        type=str,
        help='Environment to be use in the experiment'
    )
    parser.add_argument(
        '--path',
        type=str,
        help='Environment to be use in the experiment'
    )
    parser.add_argument(
        '--criteria',
        type=int,
        help='Reward threshold'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Wheter or not to print during training'
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    return args


def create_expert(
    args: argparse.Namespace, 
    model: stable_baselines.common.BaseRLModel = None, 
    env: gym.core.Env = None
) -> None:
    """
    Generate the expert data using stable-baselines generate_expert_traj.
    The file creates is a Npz File (which is a dictionary).
    The created expert data keys are as follows:

    'actions': The action executed in the 't' step
    'obs': The environment obs from the 't' step (you want to always get t and t+1).
    'rewards': The reward in the 't' step.
    'episode_returns': The final reward for each episode.
    'episode_starts': If it was the beginning of a run or not.
    """
    from stable_baselines.gail import generate_expert_traj
    from stable_baselines.common import make_vec_env
    from utils.BaselineModels import Algo
    from utils.Domains import Domain
    from utils import enjoy

    if Domain().is_continuous():
        algo = 'SAC'
    else:
        algo = 'DQN'

    if not os.path.exists('./Expert/'):
        os.makedirs('./Expert/')

    if model is None:
        model, _ = Algo().get_algo(algo)(Domain().get('name'), args.verbose)
    
    env = make_vec_env(Domain().get('name'), n_envs=10)
    model.learn(total_timesteps=1000, reset_num_timesteps=True if model is None else False)

    criteria = args.criteria if args.criteria is not None else Domain().get('criteria')

    avg_reward = enjoy(model)
    if (Domain().get('solved') or criteria) and avg_reward < criteria:
        print(f'Expert did not reach the solving criteria (Reward: {avg_reward})')
        create_expert(args, model, env)
    else:
        print(f'Expert passed the environment goal (Reward: {avg_reward})')

    if args.path is not None:
        path = args.path
    else:
        path = f'./Expert/expert_{Domain().get("name")}'

    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))

    generate_expert_traj(model, path, n_episodes=5000)

    expert = np.load(f'{path}.npz', allow_pickle=True)
    if np.mean(expert['episode_returns']) < criteria:
        print(f'Expert samples did not reach environment goal (Reward: {np.mean(expert["episode_returns"])})')
        os.remove(f'{path}.npz')
        create_expert(args, model, env)
    else:
        print(f'Expert samples have reached the environment goal (Reward: {np.mean(expert["episode_returns"])})')
        sys.exit()


def create_expert_dataset(dataset: str, amount: int, batch_size: int) -> stable_baselines.gail.ExpertDataset:
    """
    Uses the expert trajectories created by "generate_expert" and creates an Expert Dataset with a specific size.

    'dataset': str = path to the '.npz' file.
    'amount': int = how many samples in the dataset.
    'batch_size': int = size of the mini batch.
    """
    from stable_baselines.gail import ExpertDataset

    expert = ExpertDataset(
        expert_path=dataset,
        traj_limitation=-1,
        verbose=0,
        batch_size=batch_size,
        sequential_preprocessing=True
    )

    expert.train_loader.n_minibatches = amount // batch_size
    expert.val_loader.n_minibatches = amount // batch_size

    expert.train_loader.observations = expert.train_loader.observations.astype(np.float64)
    expert.observations = expert.observations.astype(np.float64)

    expert.val_loader.observations = expert.val_loader.observations.astype(np.float64)
    expert.observations = expert.observations.astype(np.float64)

    return expert


def create_random(size: int, path: str) -> None:
    import gym

    env = gym.make(Domain().get('name'))
    state = env.reset()

    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))

    dataset = np.ndarray(shape=(0, 3))
    pbar = tqdm(range(size))
    pbar.set_description_str('Creating Random')
    for _ in pbar:
        action = env.action_space.sample()
        n_state, _, done, _ = env.step(action)
        entry = np.array([state, action, n_state])[None]
        dataset = np.append(dataset, entry, axis=0)

        if done:
            state = env.reset()
        else:
            state = deepcopy(n_state)


    np.save(path, dataset)

