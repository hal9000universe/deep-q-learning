import gym


if __name__ == '__main__':
    env: gym.Env = gym.make('Humanoid-v4', render_mode='human')
    state = env.reset()
    for step in range(1000):
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        env.render()
