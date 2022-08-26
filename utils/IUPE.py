import os
import configparser

import gym
from gym.spaces import Discrete
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboard_wrapper.tensorboard import Tensorboard as Board

from .Attention import Self_Attn1D
from .Domains import Domain
from .CreateExpert import create_random


class IDM_Dataset(Dataset):
    transforms = torch.from_numpy

    def __init__(self, path, size: int = None):
        dataset = np.load(path, allow_pickle=True)
        dataset = dataset[:size, :]

        self.st = dataset[:, 0]
        self.nst = dataset[:, -1]
        self.a = dataset[:, 1]

    def __len__(self):
        return self.st.shape[0]

    def __getitem__(self, idx):

        if isinstance(self.st[idx], np.ndarray):
            st = self.transforms(self.st[idx]).float()
        elif isinstance(self.st[idx], torch.Tensor):
            st = self.st[idx]
        if isinstance(self.nst[idx], np.ndarray):
            nst = self.transforms(self.nst[idx]).float()
        elif isinstance(self.nst[idx], torch.Tensor):
            nst = self.nst[idx]
        if isinstance(self.a[idx], torch.Tensor):
            a = self.a[idx].item().float()
        else:
            a = torch.tensor(self.a[idx]).float()

        return (st, nst, a)


class Policy_Dataset(Dataset):
    
    transforms = torch.from_numpy

    def __init__(self, path, size):
        self.dataset = np.load(path, allow_pickle=True)
        self.trajectories = self.create_dataset(self.dataset, size)
        self.dataset = self.transforms(self.dataset['obs'])
        np.save(f'./tmp/expert_{Domain().given_name}', self.trajectories)

    def create_dataset(self, dataset, size):
        beggining = np.where(dataset['episode_starts'] == True)[0]
        end = np.array([*np.array(beggining - 1)[1:], dataset['episode_starts'].shape[0]])

        beggining = beggining[:size]
        end = end[:size]

        trajectories = np.ndarray(shape=(0, 3))
        for idx, (b, e) in enumerate(zip(beggining, end)):
            _b = np.array(list(range(b, e)))
            _e = (_b + 1)
            actions = dataset['actions'][_b]

            _b = _b[None].swapaxes(0, 1)
            _e = _e[None].swapaxes(0, 1)
            entry = np.hstack((_b, _e, actions))
            trajectories = np.append(trajectories, entry, axis=0)

        return trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        st, nSt, a = self.trajectories[idx].astype(int)
        st = self.dataset[st].float()
        nSt = self.dataset[nSt].float()
        a = torch.tensor(a).float()
        return st, nSt, a


class MlpAttention(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MlpAttention, self).__init__()

        out = max(8, in_dim)
        self.input = nn.Linear(in_dim, out)
        self.output = nn.Linear(out, out_dim)
        self.attention = Self_Attn1D(out, nn.LeakyReLU)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            try:
                x = torch.Tensor(x)
            except TypeError as err:
                print(err)
                print(x)
                exit()

        x = x.float()
        x = self.input(x)
        x, _ = self.attention(x)
        x = self.relu(self.output(x))
        return x


class IDM(nn.Module):
    def __init__(self, action_size, input=8):
        super().__init__()
        self.model = MlpAttention(input, action_size)

    def forward(self, state, nState):
        input = torch.cat((state, nState), 1)
        x = self.model(input)
        return x


class Policy(nn.Module):
    def __init__(self, action_size, environment, input=4):
        super(Policy, self).__init__()

        self.environment = environment
        self.model = MlpAttention(input, action_size)

    def forward(self, state: list, reduce: bool = True) -> [int, float, torch.Tensor]:
        x = self.model(state)

        if reduce:
            x = x.squeeze(0)
            if isinstance(self.environment.action_space, Discrete):
                x = int(torch.argmax(x).detach())
            else:
                x = x.detach()
        
        return x
       

class IUPE(nn.Module):
    def __init__(
        self,
        environment: gym.core.Env,
        dataset: str = None,
        device=None,
        amount: int = 5000,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.environment = environment
        self.expert_path = dataset
        self.device = device
        self.verbose = verbose
        
        self.iupe_dataset = None
        self.amount = amount
        self.batch_size = batch_size

        config = configparser.ConfigParser()
        config.read('./utils/config/iupe.ini')
        self.config = config

        folder_name = Domain().given_name[0].upper() + Domain().given_name[1:].lower()
        self.random_path = config['DEFAULT']['random_dataset'].replace('%domain%', folder_name)
        if not os.path.exists(self.random_path):
            if self.verbose:
                print('Did not find Random Dataset. Creating one...')

            size = int(config['DEFAULT']['random_dataset_size'])
            create_random(size, self.random_path)

        if isinstance(self.environment.action_space, Discrete):
            action_space = self.environment.action_space.n
            observation_space = self.environment.observation_space.shape[0]
        else:
            action_space = self.environment.action_space.shape[0]
            observation_space = self.environment.observation_space.shape[0]

        self.action_space = action_space

        self.idm = IDM(action_space, observation_space * 2)
        self.idm.to(self.device)
        self.idm_criterion = None
        if isinstance(self.environment.action_space, Discrete):
            self.idm_criterion = nn.CrossEntropyLoss()
        else:
            self.idm_criterion = nn.MSELoss()
        self.idm_optimizer = optim.Adam(self.idm.parameters(), lr=float(config['DEFAULT']['idm_lr']))
        self.random_dataset = DataLoader(
            IDM_Dataset(self.random_path),
            batch_size=batch_size,
            shuffle=True
        )

        self.policy = Policy(action_space, environment, observation_space)
        self.policy.to(self.device)
        self.policy_criterion = None
        if isinstance(self.environment.action_space, Discrete):
            self.policy_criterion = nn.CrossEntropyLoss()
        else:
            self.policy_criterion = nn.MSELoss()
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=float(config['DEFAULT']['policy_lr']))
        self.expert_dataset = DataLoader(
            Policy_Dataset(self.expert_path, self.amount),
            batch_size=batch_size,
            shuffle=True
        )

        name = str(self.environment).split('<')[-1].replace('>', '')
        self.board = Board(f'IUPE-{name}', './tmp/board/', delete=True)

    def get_env(self):
        return self.environment

    def idm_train(self):
        if self.verbose:
            self.pbar.set_description_str(desc=f'Training IDM', refresh=True)

        if not self.idm.training:
            self.idm.train()

        if self.iupe_dataset is not None:
            datasets = [self.random_dataset, self.iupe_dataset]
            names = ['random', 'alpha']
        else:
            datasets = [self.random_dataset]
            names = ['random']

        acc_t = []
        loss_t = []
        for dataset, name in zip(datasets, names):  
            for mini_batch in dataset:
                s, nS, a = mini_batch

                s = s.to(self.device)
                nS = nS.to(self.device)
                a = a.to(self.device)

                if isinstance(self.environment.action_space, Discrete):
                    a = a.long()

                self.idm_optimizer.zero_grad()
                pred = self.idm(s, nS)

                loss = self.idm_criterion(pred, a)
                loss.backward()
                self.idm_optimizer.step()
                loss_t.append(loss.item())
                
                acc = (torch.argmax(pred, 1) == a).sum().item() / a.size(0)
                acc_t.append(acc)

                if self.verbose:
                    self.pbar.update(1)

        return np.mean(acc_t), np.mean(loss_t)
            
    def policy_train(self):
        if self.verbose:
            self.pbar.set_description_str(desc="Training Policy", refresh=True)

        if not self.policy.training:
            self.policy.train()

        if self.idm.training:
            self.idm.eval()

        loss_t = []
        acc_t = []
        for mini_batch in self.expert_dataset:
            s, nS, _ = mini_batch

            s = s.to(self.device)
            nS = nS.to(self.device)

            a = self.idm(s, nS)
            action = torch.argmax(a, 1)

            self.policy_optimizer.zero_grad()
            pred = self.policy(s, reduce=False)

            loss = self.policy_criterion(pred, action)
            loss.backward()
            self.policy_optimizer.step()
            loss_t.append(loss.item())

            acc = ((torch.argmax(pred, 1) == action).sum().item() / action.shape[0])
            acc_t.append(acc)

            if self.verbose:
                self.pbar.update(1)

        return np.mean(acc_t), np.mean(loss_t)

    def create_alpha(self):
        if self.verbose:
            self.pbar.set_description_str(desc='Creating alpha', refresh=True)

        if not os.path.exists('./tmp/'):
            os.makedirs('./tmp/')

        if self.policy.training:
            self.policy.eval()

        ratio = self.generate_expert_traj(
            self.policy, 
            './tmp/alpha',
            n_episodes=1000,
            n_timesteps=self.amount,
            env=self.environment
        )

        iupe_amount = int(self.amount * ratio) + 1
        random_amount = int(len(self.random_dataset.dataset) * (1 - ratio)) + 1

        self.iupe_dataset = DataLoader(
            IDM_Dataset('./tmp/alpha.npy', iupe_amount),
            batch_size=self.batch_size,
            shuffle=True
        )

        self.random_dataset = DataLoader(
            IDM_Dataset(self.random_path, random_amount),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        return ratio

    def learn(self, total_timesteps: int = None, reset_num_timesteps: bool = False) -> None:
        
        if self.verbose:
            size = len(self.random_dataset.dataset) + len(self.expert_dataset.dataset)
            if self.iupe_dataset is not None:
                size = len(self.iupe_dataset.dataset)
            self.pbar = tqdm(range(size // self.batch_size + 1))

        # ############## IDM ############## #
        idm_acc, idm_loss = self.idm_train()
        self.board.add_scalars(
            prior='IDM',
            idm_loss=idm_loss,
            idm_acc=idm_acc
        )

        # ############## POLICY ############## #
        policy_acc, policy_loss = self.policy_train()
        self.board.add_scalars(
            prior='Policy',
            policy_loss=policy_loss,
            policy_acc=policy_acc
        )

        # ########## Create New Data ########## #
        ratio = self.create_alpha()
        self.board.add_scalars(
            prior='Alpha',
            ratio=ratio
        )

        self.board.step()
        if self.verbose:
            self.pbar.close()

    def predict(self, s):
        return self.forward(s)

    def forward(self, s, weight: str = 'MAX'):
        s = torch.Tensor(s)[None]
        s = s.to(self.device)
        pred = self.policy(s, reduce=False)

        if isinstance(self.environment.action_space, Discrete):
            if weight.upper() == 'MAX':
                return torch.argmax(pred, 1).cpu().detach().numpy().squeeze(0), None
            else:
                classes = np.arange(self.action_space)
                prob = torch.nn.functional.softmax(pred, dim=1).cpu().detach().numpy()
                a = np.random.choice(classes, p=prob[0])
                return a, None
        else:
            return pred.cpu().detach().numpy().squeeze(0), None

    def reach_goal(self, env, gym_return, total_reward):
        location, reward, _, _ = gym_return
        if 'maze' in str(env).lower():
            return True if reward == 1 else False
        elif 'cartpole' in str(env).lower():
            return True if total_reward >= 195 else False
        elif 'mountaincar' in str(env).lower():
            return True if location[0] >= 0.5 else False
        elif 'acrobot' in str(env).lower():
            return True if reward == 0 else False
        elif 'lunarlander' in str(env).lower():
            return not env.lander.awake and reward == 100
        else:
            raise NotImplementedError(f'{env} not implemented')

    def generate_expert_traj(
        self,
        policy: Policy,
        path: str, 
        n_episodes: int, 
        n_timesteps: int, 
        env: gym.core.Env
    ) -> float:
        if isinstance(env, str):
            env = gym.make(env)

        count = 0
        trajectories = []

        aer = []
        ratio = 0
        reward = 0
        timesteps = 0
        state = env.reset()
        while count < n_episodes:
            action, _ = self.forward(state, weight=self.config['DEFAULT']['weight'].upper())
            gym_return = env.step(action)
            nState, _reward, done, _ = gym_return
            reward += _reward

            trajectories.append([state, action, nState])
            timesteps += 1

            if done:
                state = env.reset()
                aer.append(reward)

                ratio += self.reach_goal(env, gym_return, reward)
                count += 1
                reward = 0
            else:
                state = nState

            if timesteps > n_timesteps:
                break
        
        trajectories = trajectories[:n_timesteps]
        np.save(path, trajectories)

        return 0 if count == 0 else ratio / count
