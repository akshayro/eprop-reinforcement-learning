import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym, time
import numpy as np
import matplotlib.pyplot as plt

from utils import EpochLogger, setup_logger_kwargs
from core import actor_critic as ac

class ReplayBuffer:
    def __init__(self, size):
        self.size, self.max_size = 0, size
        self.obs1_buf = []
        self.obs2_buf = []
        self.acts_buf = []
        self.rews_buf = []
        self.done_buf = []

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf.append(obs)
        self.obs2_buf.append(next_obs)
        self.acts_buf.append(act)
        self.rews_buf.append(rew)
        self.done_buf.append(int(done))
        while len(self.obs1_buf) > self.max_size:
            self.obs1_buf.pop(0)
            self.obs2_buf.pop(0)
            self.acts_buf.pop(0)
            self.rews_buf.pop(0)
            self.done_buf.pop(0)

        self.size = len(self.obs1_buf)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(low=0, high=self.size, size=(batch_size,))
        obs1 = torch.FloatTensor([self.obs1_buf[i] for i in idxs])
        obs2 = torch.FloatTensor([self.obs2_buf[i] for i in idxs])
        acts = torch.FloatTensor([self.acts_buf[i] for i in idxs])
        rews = torch.FloatTensor([self.rews_buf[i] for i in idxs])
        done = torch.FloatTensor([self.done_buf[i] for i in idxs])
        return [obs1, obs2, acts, rews, done]

def sac(env_name, actor_critic_function, hidden_size,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())


    replay_buffer = ReplayBuffer(replay_size)

    env, test_env = gym.make(env_name), gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = int(env.action_space.high[0])

    actor_critic = actor_critic_function(act_dim, obs_dim, hidden_size, act_limit)

    value_optimizer = optim.Adam([
        {"params":actor_critic.q1.parameters()},
        {"params":actor_critic.q2.parameters()},
        {"params":actor_critic.v.parameters()}
    ], lr)
    policy_optimizer = optim.Adam(actor_critic.policy.parameters(), lr)

    # Setup model saving
    logger.setup_pytorch_saver(actor_critic)

    start_time = time.time()

    obs, ret, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    total_ret = 0
    rewards_list = []

    for t in range(total_steps):
        if t > 50000:
            # env.render()
            # AKSHAY: commented this out so I can train on Google Colab
            print('Not rendering, on Google Colab')
        if t > start_steps:
            obs_tens = torch.from_numpy(obs).float().reshape(1,-1)
            _, act, _ = actor_critic.get_action(obs_tens)
            act = act.detach().numpy().reshape(-1)
        else:
            act = env.action_space.sample()

        obs2, ret, done, _ = env.step(act)

        total_ret += ret
        ep_ret += ret
        rewards_list.append(total_ret)
        ep_len += 1

        done = False if ep_len==max_ep_len else done

        replay_buffer.store(obs, act, ret, obs2, done)

        obs = obs2

        if done or (ep_len == max_ep_len):
            for _ in range(ep_len):
                obs1_tens, obs2_tens, acts_tens, rews_tens, done_tens = replay_buffer.sample_batch(batch_size)

                q_targ = actor_critic.compute_q_target(obs2_tens, gamma, rews_tens, done_tens)
                v_targ = actor_critic.compute_v_target(obs1_tens, alpha)

                q1_val, q2_val = actor_critic.q_function(obs1_tens, acts_tens)
                q_loss = 0.5 * (q_targ - q1_val).pow(2).mean() + 0.5 * (q_targ - q2_val).pow(2).mean()

                v_val  = actor_critic.v(obs1_tens).squeeze()
                v_loss = 0.5 * (v_targ - v_val).pow(2).mean()

                value_loss = q_loss + v_loss

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                policy_loss = -actor_critic.q_function_w_entropy(obs1_tens, alpha).mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                logger.store(LossQ=q_loss.item(), Q1Vals=q1_val.detach().numpy(), Q2Vals=q2_val.detach().numpy())
                logger.store(LossV=v_loss.item(), VVals=v_val.detach().numpy())
                logger.store(LossPi=policy_loss.item())

                actor_critic.update_target(polyak)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, ret, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # test_agent()
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True) 
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    plt.plot(rewards_list)
    plt.xlabel('steps', size=12)
    plt.ylabel('return', size=12)
    plt.savefig('sac_ant_img.png')
    plt.show()

            # Save state dict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sac(args.env, actor_critic_function=ac,
        hidden_size=[args.hid]*args.l, gamma=args.gamma, epochs=args.epochs,
        logger_kwargs=logger_kwargs)


