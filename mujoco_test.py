# On Windows: run the following before running
# set PATH=C:\Users\$username_here%\.mujoco\mujoco200\bin;%PATH%

import gym
from gym import envs

env = gym.make('Ant-v2')
# env.env.model.body_mass = env.env.model.body_mass * 10
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()