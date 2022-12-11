from agents.standard_elevator_v7_controller import StandardElevatorV7Controller
from envs.elevator_v7 import ElevatorV7Env
from stable_baselines3 import A2C, PPO
from tqdm import tqdm
from multiprocessing import Process, Pool

import torch
import numpy as np
import random


def benchmark_agent(model_filepath, num_episodes=100, num_elevators_start=1, num_elevators_end=1, num_floors_start=3, num_floors_end=3):
    RANDOM_SEED = 456
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
    all_rewards = []
    for _ in tqdm(range(num_episodes)):
        obs = env.reset(override_curriculum=True)
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
        all_rewards.append(reward_sum)

        # requests_sum += env.num_total_requests
        # dropped_sum += env.num_dropped_off

    # print(f"{model_filepath}: {dropped_sum / requests_sum: .3%}")

    all_rewards = np.array(all_rewards)
    mean = np.mean(all_rewards)
    std = np.std(all_rewards)

    print(f"{model_filepath}: {mean=}, {std=}")


def main():
    num_floors = 8
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
            "models/env_v7/elev1-1_floor3-3_rand1/9c5894.zip",
            "models/env_v7/elev1-1_floor3-3_rand2/fadd18.zip",
            # "models/env_v7/elev1-1_floor3-3_rand1/56e28e.zip",
            "Standard Controller"
        ]
    elif num_floors == 5:
        models = [
            "models/env_v7/elev1-1_floor5-5_rand0/2e083c.zip", # GOOD: non-curriculum
            "models/env_v7/elev1-1_floor5-5_rand1/d8696d.zip",
            "models/env_v7/elev1-1_floor5-5_rand2/0cacd7.zip",
            # "models/env_v7/elev1-1_floor3-5_rand0/7ac58f.zip", # GOOD: curriculum
            # "models/env_v7/elev1-1_floor3-5_rand1/2c5058.zip",
            # "models/env_v7/elev1-1_floor3-5_rand0/2c37eb.zip", # A2C
            "Standard Controller"
        ]
    elif num_floors == 8:
        models = [
            "models/env_v7/elev1-1_floor8-8_rand0/c068ad.zip", # non-curriculum 1M
            "models/env_v7/elev1-1_floor8-8_rand1/65a3a2.zip",
            "models/env_v7/elev1-1_floor8-8_rand2/490fbe.zip",
            "models/env_v7/elev1-1_floor3-8_rand0/0e7c72.zip", # curriculum 1M
            "models/env_v7/elev1-1_floor3-8_rand1/09feb9.zip",
            "models/env_v7/elev1-1_floor3-8_rand2/2a5d7c.zip",
            "Standard Controller"
        ]
    elif num_floors == 10:
        models = [
            "models/env_v7/elev1-1_floor10-10_rand0/95967f.zip", # non-curriculum
            "models/env_v7/elev1-1_floor3-10_rand0/b4d488.zip", # curriculum
            "models/env_v7/elev1-1_floor3-10_rand0/6d7dae.zip", # curriculum with reward
            "models/env_v7/elev1-1_floor3-10_rand1/08c664.zip", # long curriculum with reward
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
