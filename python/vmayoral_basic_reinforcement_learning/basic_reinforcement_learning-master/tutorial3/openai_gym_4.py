import gym
env = gym.make('CartPole-v0')
env.monitor.start('/tmp/cartpole-experiment-1',force=True)
for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        print observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break

env.monitor.close()