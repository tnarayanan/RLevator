from agents.standard_elevator_v2_controller import StandardElevatorV2Controller
from envs.elevator_v2 import ElevatorV2Env
from stable_baselines3 import A2C, PPO
from tqdm import tqdm
from multiprocessing import Process, Pool


def benchmark_agent(model_filepath, num_episodes=100, num_elevators_start=1, num_elevators_end=1, num_floors_start=3, num_floors_end=3):
    env = ElevatorV2Env(curriculum=True,
                        num_elevators_start=num_elevators_start,
                        num_elevators_end=num_elevators_end,
                        num_floors_start=num_floors_start,
                        num_floors_end=num_floors_end,
                        episode_len=300)

    if model_filepath == "Standard Controller":
        model = StandardElevatorV2Controller(env)
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
    num_floors = 5
    params = {
        'num_episodes': 100,
        'num_elevators_start': 1,
        'num_elevators_end': 1,
        'num_floors_start': num_floors,
        'num_floors_end': num_floors
    }

    if num_floors == 5:
        models = [
            # "models/env_v2/elev1-1_floor5-5/55d5b7.zip", # GOOD: non-curriculum ~55% on 5 floors
            # "models/env_v2/elev1-1_floor5-5/4a3039.zip",
            # "models/env_v2/elev1-1_floor5-5/237766.zip",
            # "models/env_v2/elev1-1_floor3-5/80f213.zip", # GOOD: curriculum ~73% on 5 floors
            "models/env_v2/elev1-1_floor3-5_rand0/2be15b.zip", # GOOD: curriculum, random seeded
            # "models/env_v2/elev1-1_floor3-5/e6e1f1.zip",
            # "models/env_v2/elev1-1_floor3-5/ee34c4.zip",
            # "Standard Controller"
        ]
    elif num_floors == 10:
        models = [
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
