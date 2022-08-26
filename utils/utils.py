from typing import TypeVar

M = TypeVar('M')

def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance


def enjoy(model: M, times: int = 100) -> float:
    import gym
    import numpy as np
    from utils.Domains import Domain

    rewards = []
    env = gym.make(Domain().get('name'))
    for _ in range(times):
        total_reward = 0
        state, done = env.reset(), False
        while not done:
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return round(float(np.mean(rewards)), 4)


