#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.saved_model import tag_constants
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

from networks import SAC
from quad_env import QuadEnv


test_episodes = 1

tf.reset_default_graph()

env = QuadEnv()
state_dim = 12
action_dim = 4
action_max = 1

save_path = os.path.join(os.getcwd(), 'saved_models/sac-quad3d_1/model.ckpt-400')
session = tf.Session()

model = SAC(session, state_dim, action_dim)
model.saver.restore(session, save_path)


# Run tests
cumulative_rewards = []
cumulative_length = []

for tep in range(test_episodes):
    
    ep_rew = 0
    ep_len = 0
    
    observation = env.reset()
    while True:
        
        if tep == 0:
            env.render()
            time.sleep(0.0001)
        
        action, _ = model.get_det_policy([observation])
        action = action[0]
        observation, reward, done, info = env.step(action)
        
        ep_rew += reward
        ep_len += 1
        
        if done: break
    
#        if test_episodes==1 and ep_rew < -100000: break
    
    cumulative_rewards.append(ep_rew)
    cumulative_length.append(ep_len)
    
episode_reward = np.mean(cumulative_rewards)
ep_len = np.mean(cumulative_length)

extras = {'rew_max' : np.max(cumulative_rewards),
          'rew_min' : np.min(cumulative_rewards),
          'len_max' : np.max(cumulative_length),
          'len_min' : np.min(cumulative_length)}


print("Reward - avg: {:.1f}, max: {:.1f}, min: {:.1f}".format(
        episode_reward, extras['rew_max'], extras['rew_min']))
print("Length - avg: {}, max: {}, min: {}".format(
        ep_len, extras['len_max'], extras['len_min']))
