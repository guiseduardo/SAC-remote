#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import shutil

from networks import ReplayMemory, SAC, MetricsTracker
from quad_env import QuadEnv


#%% Training config

# Environment
env = QuadEnv()
state_dim = 12
action_dim = 4
action_max = 1

# Hyperparameters
lr_a = 5e-4
lr_c = 5e-4
lr_t = 5e-4
tau = 0.005
gamma = 0.99
h = -1     # Minimum expected entropy (Haarnoja et al.) Def.: -action_dim
log_std_min = -20
log_std_max = 2      # Def.: -action_max
buffer_size = 2000000

# Training steps
init_steps = 10000
episodes_max = 2000
gradient_steps = 100
update_every = 50
batch_size = 800
test_episodes = 10

# Runtime config
int_render = 0  # How many episodes to wait for a render
render_step = 0.001
save_path = os.path.join(os.getcwd(), 'saved_models/sac-quad3d_1/model.ckpt')
save_every = 100  # Save model checkpoints


#%% Start
t_run = time.time()

# Set up Tensorflow
session = tf.Session()

# Create networks
model = SAC(session, state_dim, action_dim,
              lr_a=lr_a, lr_c=lr_c, lr_t=lr_t,
              tau=tau, h=h, a_max=action_max,
              log_std_min=log_std_min, log_std_max=log_std_max)

# Initialize weights
session.run(tf.global_variables_initializer())

# Initialize replay memory
memory = ReplayMemory(max_size=buffer_size)

update_count = 0
#solved_tracker = np.zeros(100)
tracker = MetricsTracker(model)

for ep in range(episodes_max):

    observation = env.reset()

    t_start = time.time()
    state_track = []
    action_track = []

#%% Run an episode
    while True:

        # Choose action
        if memory.size() >= init_steps:
            action, _ = model.get_action([observation])
            action = action[0]
        else:
            action = np.random.uniform(-action_max,action_max,action_dim)

        # Take step, observate s', r
        prev_obs = observation
        observation, reward, done, info = env.step(action)

        # Add to replay memory
        memory.add(prev_obs, action, reward, observation, done)

        state_track.append(observation)
        action_track.append(action)

        if done:
            if memory.size() >= init_steps:
                break
            else:
                episode_reward = 0
                observation = env.reset()
                t_start = time.time()
                ep_len = 0
                state_track = []
                action_track = []

#%% Train offline
    if memory.size() >= init_steps:
        for j in range(gradient_steps):

            # Sample minibatch
            s, a, r, sp, d = memory.sample_minibatch(batch_size)

            # Compute targets for the Q functions
            targets = model.get_target(sp)
            y = [r[i] + gamma*(1-d[i])*targets[i] for i in range(len(d))]

            # Train model
            loss_q, loss_pi = model.train(s, a, y)

            # Update target networks
            update_count += 1
            if update_count%update_every:
                model.update()

#%% Test policy
    if memory.size() >= init_steps:

        cumulative_rewards = []
        cumulative_length = []
        initial_states = []

        for tep in range(test_episodes):

            ep_rew = 0
            ep_len = 0

            observation = env.reset()
            initial_states.append(observation)
            while True:

                if int_render!=0 and tep==0 and (ep+1)%int_render==0:
                    env.render()
                    time.sleep(render_step)

                action, _ = model.get_det_policy([observation])
                action = action[0]
                observation, reward, done, info = env.step(action)

                ep_rew += reward
                ep_len += 1

                if done: break

            cumulative_rewards.append(ep_rew)
            cumulative_length.append(ep_len)

        episode_reward = np.mean(cumulative_rewards)
        ep_len = np.mean(cumulative_length)
        init_values = model.get_init_value(np.array(initial_states))

        extras = {'rew_max' : np.max(cumulative_rewards),
                  'rew_min' : np.min(cumulative_rewards),
                  'len_max' : np.max(cumulative_length),
                  'len_min' : np.min(cumulative_length),
                  'init_st' : init_values}

#%% Display metrics
    print("Episode {} - Ep.len.: {}, Rew.: {:.3f}, Temp.: {:.3f}, Elapsed: {:.3f} s".format(
            ep+1, ep_len, episode_reward, model.get_alpha(), time.time()-t_start))

    tracker.add(memory.sample_minibatch(100)[3], episode_reward, ep_len, extras)

    if (ep+1)%save_every == 0:
#        tracker.plot()
        model.save_weigths(save_path, ep+1)
        tracker.dump('./cache')


#tracker.plot()
#plt.draw()
model.save_weigths(save_path, ep+1)
tracker.dump('./cache')
print("Total time elapsed: {:.1f} s".format(time.time()-t_run))

