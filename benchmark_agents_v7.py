from agents.standard_elevator_v7_controller import StandardElevatorV7Controller
from envs.elevator_v7 import ElevatorV7Env
from stable_baselines3 import A2C, PPO
from tqdm import tqdm
from multiprocessing import Process, Pool

import torch
import numpy as np
import random


def benchmark_agent(model_filepath, num_episodes=100, num_elevators_start=1, num_elevators_end=1, num_floors_start=3, num_floors_end=3):
    RANDOM_SEED = 123
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    env = ElevatorV7Env(curriculum=True,
                        num_elevators_start=num_elevators_start,
                        num_elevators_end=num_elevators_end,
                        num_floors_start=num_floors_start,
                        num_floors_end=num_floors_end,
                        episode_len=100,
                        random_seed=RANDOM_SEED)

    if model_filepath == "Standard Controller":
        model = StandardElevatorV7Controller(env)
    else:
        model = PPO.load(model_filepath, env=env)

    print(f"Benchmarking {model_filepath}")

    # requests_sum = 0
    # dropped_sum = 0
    reward_sum = 0
    for _ in tqdm(range(num_episodes)):
        obs = env.reset(override_curriculum=True)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward

        # requests_sum += env.num_total_requests
        # dropped_sum += env.num_dropped_off

    # print(f"{model_filepath}: {dropped_sum / requests_sum: .3%}")
    print(f"{model_filepath}: {reward_sum / num_episodes}")


def main():
    num_floors = 10
    params = {
        'num_episodes': 100,
        'num_elevators_start': 1,
        'num_elevators_end': 1,
        'num_floors_start': num_floors,
        'num_floors_end': num_floors
    }
    if num_floors == 3:
        models = [
            "models/env_v7/elev1-1_floor3-3_rand0/16e031.zip", # GOOD: non-curriculum
            # "models/env_v7/elev1-1_floor3-3_rand1/56e28e.zip",
            "Standard Controller"
        ]
    elif num_floors == 5:
        models = [
            "models/env_v7/elev1-1_floor5-5_rand0/2e083c.zip", # GOOD: non-curriculum
            # "models/env_v7/elev1-1_floor5-5_rand1/043c7e.zip",
            # "models/env_v7/elev1-1_floor3-5_rand0/7ac58f.zip", # GOOD: curriculum
            # "models/env_v7/elev1-1_floor3-5_rand1/2c5058.zip",
            "Standard Controller"
        ]
    elif num_floors == 10:
        models = [
            "models/env_v7/elev1-1_floor10-10_rand0/95967f.zip", # non-curriculum
            "models/env_v7/elev1-1_floor3-10_rand0/b4d488.zip", # curriculum
            "models/env_v7/elev1-1_floor3-10_rand0/6d7dae.zip", # curriculum with reward
            "Standard Controller"
        ]
    else:
        raise AssertionError(f"models not defined for {num_floors} floors")

    procs = []

    for model_name in models:
        proc = Process(target=benchmark_agent, args=(model_name,), kwargs=params)
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    main()
