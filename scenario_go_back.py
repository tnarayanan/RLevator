from envs.elevator_v5_go_back import ElevatorV5GoBackEnv
from agents.standard_elevator_v5_go_back_controller import StandardElevatorV5GoBackController
from agents.standard_elevator_v7_controller import StandardElevatorV7Controller
from stable_baselines3 import A2C, PPO
from envs.elevator_base import Request

VERBOSE = 0
RANDOM_SEED = 0

NUM_ELEVATORS_START = 1
NUM_ELEVATORS_END = 1
NUM_FLOORS_START = 8
NUM_FLOORS_END = 8

env = ElevatorV5GoBackEnv(curriculum=True,
                          num_elevators_start=NUM_ELEVATORS_START,
                          num_elevators_end=NUM_ELEVATORS_END,
                          num_floors_start=NUM_FLOORS_START,
                          num_floors_end=NUM_FLOORS_END,
                          episode_len=300,
                          random_seed=RANDOM_SEED)
#
# env.elevators[0].floor = 1
# env.elevators[0].state = 0
# env.elevators[0].time_to_next_floor = 5
# env.elevators[0].add_request(Request(0, 0))
# env.unassigned_requests = {2: [Request(0, 0)]}
# env.num_total_requests += 2

# model = StandardElevatorV5GoBackController(env)
# model = StandardElevatorV7Controller(env)
model = PPO.load("models/env_v7/elev1-1_floor3-8_rand0/0e7c72.zip", env=env)
obs = env.get_obs()
import ipdb
for i in range(100):
    ipdb.set_trace()
    action, _ = model.predict(obs, deterministic=True)
    print(f"target floor: {action[0]}")
    obs, reward, done, _ = env.step(action)
    print(env.elevators[0])
    print(env.unassigned_requests)



