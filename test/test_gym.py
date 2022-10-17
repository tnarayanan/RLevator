import gym
env = gym.make('LunarLander-v2')
# env.action_space.seed(42)

obs = env.reset()

for _ in range(1000):
    obs, reward, done, info = env.step(env.action_space.sample())

    if done:
        obs = env.reset()

    env.render()

env.close()