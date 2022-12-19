from envs.elevator_v7 import ElevatorV7Env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
import secrets
import torch
import numpy as np
import random
import argparse


class TensorboardCallback(BaseCallback):
    def __init__(self, caller_env, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = caller_env

    def _on_step(self) -> bool:
        self.logger.record("env/num_floors", self.env.num_floors)
        self.logger.record("env/num_elevators", self.env.num_elevators)
        if sum(self.env.requests_history) > 0:
            self.logger.record("env/drop_history", sum(self.env.dropped_off_history) / sum(self.env.requests_history))
        return True


def main(args):
    num_elevators_start = 1
    num_elevators_end = 1
    num_floors_start = int(args.num_floors_start)
    num_floors_end = int(args.num_floors_end)

    total_timesteps = int(args.timesteps)
    verbose = int(args.verbose)

    # generate model identifier before resetting seeds
    model_identifier = secrets.token_hex(3)

    random_seed = int(args.seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    env_identifier = f"env_v7/elev{num_elevators_start}-{num_elevators_end}_floor{num_floors_start}-{num_floors_end}_rand{random_seed}"

    print(f"Training {env_identifier}/{model_identifier} for {total_timesteps} timesteps")

    tensorboard_dir = f"./tensorboard/{env_identifier}/{model_identifier}/"

    env = ElevatorV7Env(curriculum=True,
                        num_elevators_start=num_elevators_start,
                        num_elevators_end=num_elevators_end,
                        num_floors_start=num_floors_start,
                        num_floors_end=num_floors_end,
                        episode_len=100,
                        random_seed=random_seed)

    model = PPO("MlpPolicy", env, verbose=verbose, tensorboard_log=tensorboard_dir)
    model.learn(total_timesteps=total_timesteps, callback=TensorboardCallback(env))
    model.save(f"./models/{env_identifier}/{model_identifier}")

    # test the trained model for 2000 timesteps
    # for full testing, see benchmark_agents.py

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_floors_start', '-nfs', default=3, help='Number of floors at start')
    parser.add_argument('--num_floors_end', '-nfe', default=3, help='Number of floors at end')
    parser.add_argument('--timesteps', '-t', default=100_000, help='Number of timesteps to train')
    parser.add_argument('--seed', '-s', default=0, help='Random seed to use')
    parser.add_argument('--verbose', '-v', default=0, help='Verbosity (0 or 1)')

    args = parser.parse_args()
    main(args)
