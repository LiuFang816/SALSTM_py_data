import time

'''
Evaluate a policy over one trajectory

Params:
 - env: the environement
 - policy: the policy to evaluate
 - maximum_episode_length: the maximum length of the trajectory
 - discount_factor: the discount_factor
 - render: if True, then render is called during the evaluation
'''
def rl_evaluate_policy(env,policy,maximum_episode_length=100000,discount_factor=1.0,render=False):
    reward=0
    df=1.0
    policy.start_episode()
    observation = env.reset()
    if (render):
        env.render()
    policy.observe(observation)
    for t in range(maximum_episode_length):
        action = policy.sample()
        observation, immediate_reward, finished, info = env.step(action)
        if (render):
            env.render()
        reward=reward+df*immediate_reward
        df=df*discount_factor
        if (finished):
            break
        policy.observe(observation)

    policy.end_episode()
    return reward

'''
Evaluate a policy over multiple trajectories

Params:
 - env: the environement
 - policy: the policy to evaluate
 - maximum_episode_length: the maximum length of the trajectory
 - discount_factor: the discount_factor
 - nb_episodes: the number of episodes to sample
'''
def rl_evaluate_policy_multiple_times(env,policy,maximum_episode_length=100000,discount_factor=1.0,nb_episodes=1,render=False):
    r=0

    for i in range(nb_episodes):

        re=False
        if (i==0):
            re=True
        if (render==False):
            re=False

        r=r+rl_evaluate_policy(env,policy,maximum_episode_length=maximum_episode_length,discount_factor=discount_factor,render=re)

    return r/nb_episodes
