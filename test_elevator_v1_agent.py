from envs.elevator_v1 import ElevatorV1Env
from agents.standard_elevator_v1_controller import StandardElevatorV1Controller
from stable_baselines3 import A2C, PPO

env = ElevatorV1Env(num_elevators=1, num_floors=3, episode_len=300)

# 1 elev 3 floors, ep len 300, 150k, +100 rew on success
# ppo_mlp/PPO_2
# a2c_mlp/A2C_9

# 1 elev 3 floors, ep len 300, 150k, 0 rew on success
# ppo_mlp/PPO_3
# a2c_mlp/A2C_10


# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/ppo_mlp/")
# model.learn(total_timesteps=1_000_000)
model = StandardElevatorV1Controller(env)

obs = env.reset()
total_reward = 0
for i in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        print(f"{total_reward = }")
        print(f"    {env.num_total_requests = }")
        print(f"    {env.num_dropped_off = }")
        total_reward = 0
        obs = env.reset()
