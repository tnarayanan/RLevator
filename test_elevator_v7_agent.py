from envs.elevator_v7 import ElevatorV7Env
from agents.standard_elevator_v7_controller import StandardElevatorV7Controller
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
import secrets
import torch
import numpy as np
import random

# generate model identifier before resetting seeds
model_identifier = secrets.token_hex(3)

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


NUM_ELEVATORS_START = 1
NUM_ELEVATORS_END = 1
NUM_FLOORS_START = 3
NUM_FLOORS_END = 10

TOTAL_TIMESTEPS = 1_000_000
VERBOSE = 0

env_identifier = f"env_v7/elev{NUM_ELEVATORS_START}-{NUM_ELEVATORS_END}_floor{NUM_FLOORS_START}-{NUM_FLOORS_END}_rand{RANDOM_SEED}"

print(f"Training {env_identifier}/{model_identifier} for {TOTAL_TIMESTEPS} timesteps")

tensorboard_dir = f"./tensorboard/{env_identifier}/{model_identifier}/"

env = ElevatorV7Env(curriculum=True,
                    num_elevators_start=NUM_ELEVATORS_START,
                    num_elevators_end=NUM_ELEVATORS_END,
                    num_floors_start=NUM_FLOORS_START,
                    num_floors_end=NUM_FLOORS_END,
                    episode_len=100,
                    random_seed=RANDOM_SEED)


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        # self.env = caller_env

    def _on_step(self) -> bool:
        self.logger.record("env/num_floors", env.num_floors)
        self.logger.record("env/num_elevators", env.num_elevators)
        if sum(env.requests_history) > 0:
            self.logger.record("env/drop_history", sum(env.dropped_off_history) / sum(env.requests_history))
        return True


# policy_kwargs = dict(activation_fn=torch.nn.Tanh,
#                      net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])])
model = PPO("MlpPolicy", env, verbose=VERBOSE, tensorboard_log=tensorboard_dir)
# print(model.policy)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=TensorboardCallback())
model.save(f"./models/{env_identifier}/{model_identifier}")
# model = PPO.load(f"./models/{env_identifier}/ac2701.zip", env=env)
# model = StandardElevatorV7Controller(env)

obs = env.reset(override_curriculum=True)
total_reward = 0
for i in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        print(f"{env.num_floors}: {total_reward = }")
        print(f"    {env.num_total_requests = }")
        print(f"    {env.num_dropped_off = }")
        total_reward = 0
        obs = env.reset(override_curriculum=True)
