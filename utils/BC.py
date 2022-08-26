from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset

class BC:
    def __init__(self, env, dataset, amount, batch_size, verbose, **kwargs) -> None:
        self.env = env
        self.dataset_path = dataset
        self.amount = amount
        self.batch_size = batch_size
        self.verbose = verbose

        self.dataset = ExpertDataset(
            dataset, 
            traj_limitation=amount, 
            verbose=verbose
        )
        self.model = GAIL('MlpPolicy', self.env, self.dataset, verbose=verbose)

    def learn(self, **kwargs):
        self.model.pretrain(self.dataset, n_epochs=1)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.model.predict(observation)