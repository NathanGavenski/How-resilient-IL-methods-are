from typing import TypeVar

import gym

from utils.config.DomainInfo import domain
from utils.utils import singleton


def get_all_domains() -> list:
    """
    Return all domains.
    """
    return list(domain.keys())


T = TypeVar('T')


@singleton
class Domain:
    def __init__(self, name: str = None) -> None:
        self.given_name = name.lower()
        self.domain = domain[self.given_name]

    def get(self, item: str) -> T:
        return self.domain[item]

    def get_domain(self, name: str):
        self.given_name = name.lower()
        self.domain = domain[self.given_name]
        return self

    def is_solved(self) -> bool:
        return self.domain['solved']

    def is_continuous(self) -> bool:
        env_name = self.domain.get('name')
        env = gym.make(env_name)
        return isinstance(env.action_space, gym.spaces.Box)
