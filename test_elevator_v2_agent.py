from envs.elevator_v2 import ElevatorV2Env
from agents.standard_elevator_controller import StandardElevatorController
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback

NUM_ELEVATORS_START = 1
NUM_ELEVATORS_END = 1
NUM_FLOORS_START = 3
NUM_FLOORS_END = 10

TOTAL_TIMESTEPS = 10_000
VERBOSE = 0

tensorboard_dir = f"./tensorboard/test_env_v2/elev{NUM_ELEVATORS_START}-{NUM_ELEVATORS_END}_floor{NUM_FLOORS_START}-{NUM_FLOORS_END}/"

env = ElevatorV2Env(curriculum=True,
                    num_elevators_start=NUM_ELEVATORS_START,
                    num_elevators_end=NUM_ELEVATORS_END,
                    num_floors_start=NUM_FLOORS_START,
                    num_floors_end=NUM_FLOORS_END,
                    episode_len=300)


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


model = PPO("MlpPolicy", env, verbose=VERBOSE, tensorboard_log=tensorboard_dir)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=TensorboardCallback())
# model = StandardElevatorController(env)

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
