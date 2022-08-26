# How Resilient are Imitation Learning Methods to Sub-Optimal Experts?

Official implementation for the results of [How Resilient are Imitation Learning Methods to Sub-Optimal Experts?]()

---

Imitation Learning (IL) algorithms try to mimic expert behavior in order to be capable of performing specific tasks, but it remains unclear what those strategies could achieve when learning from sub-optimal data (faulty experts). Studying how Imitation Learning approaches learn when dealing with different degrees of quality from the observations can benefit tasks such as optimizing data collection, producing interpretable models, reducing bias when using sub-optimal experts, and more. Therefore, in this work we provide extensive experiments to verify how different Imitation Learning agents perform under various degrees of expert optimality. We experiment with four IL algorithms, three of them that learn self-supervisedly and one that uses the ground-truth labels (BC) in four different environments (tasks), and we compare them using optimal and sub-optimal experts. For assessing the performance of each agent, we compute two metrics: Performance and Average Episodic Reward. Our experiments show that IL approaches that learn self-supervisedly are relatively resilient to sub-optimal experts, which is not the case of the supervised approach. We also observe that sub-optimal experts are sometimes beneficial since they seem to act as a kind of regularization method, preventing models from data overfitting.

---

## Dependencies

All dependencies are also listed at `./requirements/requirements.txt`.

```
# Envs and experts
stable-baselines3[extra]
git+https://github.com/hill-a/stable-baselines
gym==0.21.0

# Dependencies
mpi==1.0.0
mpi4py==3.1.3
numpy==1.20
box2d-py

# Utilities
tensorboard
tensorboard-wrapper
tqdm
gdown

# For tensorflow models
tensorflow-gpu==1.15

# For pytorch models
--find-links https://download.pytorch.org/whl/torch_stable.html
torch==1.7.0+cu110 
torchvision==0.8.1+cu110 
torchaudio==0.7.0
```
---

## Downloading the data

You can download the data we use to train each model [here](). \
Or use the script in `./scrpts/download.sh` (*It uses `gdown` as a dependency*).

---

## Creating Experts

If, for some reason, the data is not available, you can create the same data as we did by using the following:

```{bash}
python create_expert.py --domain <DOMAIN NAME> --criteria <CRITERIA> --path <PATH>
```

DOMAIN NAME: Follows the nomenclature from `./utils/config/DomainInfo.py`. \
CRITERIA: Is the reward threshold you want to stop training your expert agent. \
PATH: Where you want to save your expert data.

---

## Training each algorithm

For training each algorithm you can use the following command line:

```{bash}
python il_experiments.py \
    --gpu -1 \
    --data ./Expert/expert_MountainCar-v0.npz \
    --algo BC \
    --exp 10 \
    --epochs 100 \
    --domain mountaincar \
    --timesteps 1000
```

---

## Citation

```{latex}
@inproceedings{gavenski2022how,
  title={How Resilient are Imitation Learning Methods to Sub-Optimal Experts?},
  author={Nathan Gavenski and Juarez Monteiro and Adilson Medronha and Rodrigo Barros},
  booktitle={2022 Brazilian Conference on Intelligent Systems (BRACIS)},
  year={2022},
  organization={IEEE}
}
```