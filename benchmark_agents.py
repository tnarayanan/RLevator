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

    requests_sum = 0
    dropped_sum = 0
    for _ in tqdm(range(num_episodes)):
        obs = env.reset(override_curriculum=True)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

        requests_sum += env.num_total_requests
        dropped_sum += env.num_dropped_off

    print(f"{model_filepath}: {dropped_sum / requests_sum: .3%}")


def main():
    params = {
        'num_episodes': 100,
        'num_elevators_start': 1,
        'num_elevators_end': 1,
        'num_floors_start': 5,
        'num_floors_end': 5
    }

    models = [
        "models/env_v2/elev1-1_floor5-5/55d5b7.zip",
        "models/env_v2/elev1-1_floor5-5/4a3039.zip",
        "models/env_v2/elev1-1_floor5-5/237766.zip",
        "Standard Controller"
    ]

    procs = []

    for model_name in models:
        proc = Process(target=benchmark_agent, args=(model_name,), kwargs=params)
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    main()
