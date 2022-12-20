from envs.elevator_v5_double_back import ElevatorV5GoBackEnv
from agents.standard_elevator_v5_double_back_controller import StandardElevatorV5GoBackController
from agents.standard_elevator_v7_controller import StandardElevatorV7Controller
from stable_baselines3 import A2C, PPO
from envs.elevator_base import Request

# for animation
from elevator_animation import ElevatorAnimation

VERBOSE = 0
RANDOM_SEED = 0

NUM_ELEVATORS_START = 1
NUM_ELEVATORS_END = 1
NUM_FLOORS_START = 5
NUM_FLOORS_END = 5

env = ElevatorV5GoBackEnv(curriculum=True,
                          num_elevators_start=NUM_ELEVATORS_START,
                          num_elevators_end=NUM_ELEVATORS_END,
                          num_floors_start=NUM_FLOORS_START,
                          num_floors_end=NUM_FLOORS_END,
                          episode_len=300,
                          random_seed=RANDOM_SEED)


# model = StandardElevatorV5GoBackController(env)
# model = StandardElevatorV7Controller(env)
# model = PPO.load("models/env_v7/elev1-1_floor3-3_rand1/dd23f6.zip", env=env)
model = PPO.load("models/env_v7/elev1-1_floor5-5_rand0/9b9f8c.zip", env=env)
obs = env.get_obs()


animation = ElevatorAnimation(env, title="RLevator Double-Back Scenario")

for i in range(100):
    animation.set_environment(env)
    animation.draw_environment()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)


