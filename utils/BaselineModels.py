from telnetlib import DO
from tkinter import W
import types
from typing import Union, TypeVar

import numpy as np
import gym
import stable_baselines
import tensorflow as tf
import torch

from .Domains import Domain
from utils import singleton


class ActionSpaceException(Exception):
    pass


def get_env(domain: Domain) -> gym.core.Env:
    env = gym.make(domain.get('name'))

    if domain.get('type') == 'BARK':
        if 'highway' in domain.get('name'):
            from bark_ml.environments.gym import DiscreteHighwayGym
            high = env.observation_space.high
            low = env.observation_space.low
            DiscreteHighwayGym.observation_space = gym.spaces.box.Box(low, high, dtype=np.float32)

    return env


def TRPO(domain: str, verbose: bool = True) -> Union[stable_baselines.TRPO, gym.core.Env]:
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines import TRPO

    env = get_env(domain)
    with tf.Graph().as_default():
        _verbose = 1 if verbose else 0
        model = TRPO(MlpPolicy, env, verbose=verbose)

    return model, env


def DDPG(environment_name: str, verbose: bool = True) -> Union[stable_baselines.DDPG, gym.core.Env]:
    from stable_baselines.ddpg.policies import MlpPolicy
    from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
    from stable_baselines import DDPG

    env = gym.make(environment_name)
    if isinstance(env.action_space, gym.spaces.Discrete):
        raise ActionSpaceException('DDPG: Discrete action space not suported.')

    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    with tf.Graph().as_default():
        _verbose = 1 if verbose else 0
        model = DDPG(MlpPolicy, env, verbose=_verbose, param_noise=param_noise, action_noise=action_noise)

    return model, env


def SAC(environment_name: str, verbose: bool = True) -> Union[stable_baselines.SAC, gym.core.Env]:
    from stable_baselines.sac.policies import MlpPolicy
    from stable_baselines import SAC

    env = gym.make(environment_name)
    if isinstance(env.action_space, gym.spaces.Discrete):
        raise ActionSpaceException('SAC: Discrete action space not suported.')

    with tf.Graph().as_default():
        _verbose = 1 if verbose else 0
        model = SAC(MlpPolicy, env, verbose=_verbose)

    return model, env


def DQN(domain: Domain, verbose: bool = True) -> Union[stable_baselines.DQN, gym.core.Env]:
    from stable_baselines.deepq.policies import MlpPolicy
    from stable_baselines import DQN

    env = get_env(domain)
    if isinstance(env.action_space, gym.spaces.Box):
        raise ActionSpaceException('DQN: Box action space not suported.')

    with tf.Graph().as_default():
        _verbose = 1 if verbose else 0
        model = DQN(MlpPolicy, env, verbose=_verbose)

    return model, env


def ACER(environment_name: str, verbose: bool = True) -> Union[stable_baselines.ACER, gym.core.Env]:
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines import ACER

    env = gym.make(environment_name)
    if isinstance(env.action_space, gym.spaces.Box):
        raise ActionSpaceException('ACER: Box action space not suported.')

    with tf.Graph().as_default():
        _verbose = 1 if verbose else 0
        model = ACER(MlpPolicy, env, verbose=_verbose)
    
    return model, env


def HER(environment_name):
    raise NotImplementedError()


def GAIL(
        environment_name: str,
        dataset: bool = None,
        amount: int = 5000,
        batch_size: int = 1,
        verbose: bool = True,
        **kwargs
) -> Union[stable_baselines.GAIL, gym.core.Env]:
    from stable_baselines import GAIL
    from stable_baselines.gail import ExpertDataset

    if (amount / batch_size) % 2 > 0:
        raise Exception(f"Batch size incompatable: batch_size = {amount / batch_size}")

    expert = ExpertDataset(
        expert_path=dataset,
        traj_limitation=-1,
        verbose=0,
        batch_size=batch_size
    )

    expert.train_loader.n_minibatches = amount // batch_size
    expert.val_loader.n_minibatches = amount // batch_size

    expert.train_loader.observations = expert.train_loader.observations.astype(np.float64)
    expert.observations = expert.observations.astype(np.float64)

    expert.val_loader.observations = expert.val_loader.observations.astype(np.float64)
    expert.observations = expert.observations.astype(np.float64)

    env = gym.make(environment_name)
    with tf.Graph().as_default():
        _verbose = 1 if verbose else 0
        model = GAIL('MlpPolicy', environment_name, expert, verbose=_verbose)

    return model, env


def BC(
    environment_name: str,
    dataset: str = None,
    amount: int = 5000,
    batch_size: int = 1,
    verbose: bool = True,
    **kwargs,
) -> Union[stable_baselines.GAIL, gym.core.Env]:
    from .BC import BC

    env = gym.make(environment_name)
    model = BC(
        environment_name,
        dataset,
        amount,
        batch_size,
        verbose
    )

    return model, env


def IUPE(
        environment_name: str,
        device: torch.device = None,
        dataset: str = None,
        amount: int = 5000,
        batch_size: int = 1,
        verbose: bool = True,
        **kwargs
) -> Union[torch.nn.Module, gym.core.Env]:
    from .IUPE import IUPE

    env = gym.make(environment_name)
    model = IUPE(
        environment=env,
        dataset=dataset,
        device=device,
        amount=amount,
        batch_size=batch_size,
        verbose=verbose
    )

    return model, env


def ILPO(
        environment_name: str,
        batch_size: int = 1,
        verbose: bool = False,
        **kwargs
) -> Union[object, gym.core.Env]:
    from .ILPO import ILPO

    env = gym.make(environment_name)
    model = ILPO(environment_name, batch_size, verbose)

    return model, env


algos = {
    'DQN': {
        'method': DQN,
        'continuous': False
    },
    'TRPO': {
        'method': TRPO,
        'continuous': False
    },
    'DDPG': {
        'method': DDPG,
        'continuous': True
    },
    'SAC': {
        'method': SAC,
        'continuous': True
    },
    'ACER': {
        'method': ACER,
        'continuous': False
    },
    'BC': {
        'method': BC,
        'continuous': 'BOTH'
    },
    'GAIL': {
        'method': GAIL,
        'continuous': 'BOTH'
    },
    'BCO': {
        'method': None,
        'continuous': 'BOTH'
    },
    'ABCO': {
        'method': None,
        'continuous': 'BOTH'
    },
    'IUPE': {
        'method': IUPE,
        'continuous': 'BOTH'
    },
    'ILPO': {
        'method': ILPO,
        'continuous': False
    },
    'MOBILE': {
        'method': None,
        'continuous': False
    },
}

F = TypeVar('F')


@singleton
class Algo:

    def get_algo(self, name: str) -> F:
        return algos[name.upper()]['method']

    def get_all_algos(self) -> list:
        return list(algos.keys())

    def get_all_algos_by_type(self, type: str) -> list:
        if type.lower() not in ['il', 'rl']:
            raise Exception('Type should be in ["rl", "il"]')

        # TODO add BCO, ABCO and MOBILE
        if type.lower() == 'il':
            return ['BC', 'GAIL', 'IUPE', 'ILPO', 'ALL']
        elif type.lower() == 'rl':
            return ['DQN', 'TRPO', 'DDPG', 'SAC', 'HER', 'ALL']
        else:
            return []

    def get_all_algos_by_domain(self, continuous: bool) -> list:
        result = []
        for key, value in algos.items():
            if value['continuous'] == continuous or value['continuous'] == 'BOTH':
                result.append(key)
        return result

    def get_all_algos_by_domain_and_type(self, type: str, continuous: bool) -> list:
        domain = self.get_all_algos_by_domain(continuous)
        type = self.get_all_algos_by_type(type)[:-1]
        return list(set(domain).intersection(type))
