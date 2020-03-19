#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1, keepdims=True)

#%% Memory Buffer
class ReplayMemory():
   def __init__(self, max_size=1000000):
      self.memory = []
      self.max_size = max_size

   def sample_minibatch(self, batch_size):
      batch = [self.memory[np.random.randint(0, len(self.memory)-1)] for a in range(batch_size)]
      obs = [e[0] for e in batch]
      act = [e[1] for e in batch]
      rew = [e[2] for e in batch]
      nobs = [e[3] for e in batch]
      done = [e[4] for e in batch]
      return obs, act, rew, nobs, done

   def add(self, obs, act, rew, nobs, done):
      self.memory.append((obs, act, rew, nobs, done))

      # Added to limit buffer size
#      if len(self.memory) > self.max_size: del self.memory[0]

   def size(self):
      return len(self.memory)


#%% SAC object
class SAC():
    def __init__(self, sess, s_dim, a_dim,
                 lr_a=3e-4, lr_c=3e-4, lr_t=3e-4,
                 tau=0.001, log_std_min=-20, log_std_max=2,
                 a_max=1.0, h=None):

        self.session = sess
        self.layer1_size = 64
        self.layer2_size = 64
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        if h is None: h = -a_dim
        self.perf_track = []
        self.perf_track_ls = []

        # Temperature - automatically adjusted (Haarnoja et al.)
        self.log_alpha = tf.get_variable(name='log_alpha', dtype=tf.float32, initializer=0.)
        self.alpha = tf.exp(self.log_alpha)

        # Build networks
        self.s_in = tf.placeholder(tf.float32, shape=(None,s_dim), name='s_in')
        prev_vars = len(tf.trainable_variables())

        #%% Actor network
        actor_hl1 = tf.layers.dense(self.s_in, self.layer1_size, activation=tf.nn.relu, name='actor_hl1')
        actor_hl2 = tf.layers.dense(actor_hl1, self.layer2_size, activation=tf.nn.relu, name='actor_hl2')

        actor_mu_hl1 = tf.layers.dense(actor_hl2, 32, activation=tf.nn.relu, name='actor_mu_hl1')
        actor_mu_hl2 = tf.layers.dense(actor_mu_hl1, 32, activation=tf.nn.relu, name='actor_mu_hl2')

        actor_sig_hl1 = tf.layers.dense(actor_hl2, 32, activation=tf.nn.relu, name='actor_sig_hl1')
        actor_sig_hl2 = tf.layers.dense(actor_sig_hl1, 32, activation=tf.nn.relu, name='actor_sig_hl2')

        self.mu = tf.layers.dense(actor_mu_hl2, a_dim, activation=None, name='mu')
        self.log_std = tf.layers.dense(actor_sig_hl2, a_dim, activation=tf.sigmoid, name='log_std')

        self.a_vars = tf.trainable_variables()[prev_vars:]

        # Reparametrization trick
        self.log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (self.log_std + 1)
        self.noise = tf.exp(self.log_std) * tf.random.normal(shape=tf.shape(self.mu))
        self.pi = self.mu + self.noise
        self.log_prob = gaussian_likelihood(self.pi, self.mu, self.log_std)

        # Squashing
        self.mu = tf.tanh(self.mu)
        self.pi = tf.tanh(self.pi)
        self.log_prob -= tf.reduce_sum(tf.log(1 - self.pi**2 + 1e-6), axis=1, keepdims=True)

        # Action scaling
        self.mu = a_max*self.mu
        self.pi = a_max*self.pi

        #%% Critic network
        self.a_in = tf.placeholder_with_default(self.pi, shape=(None,a_dim), name='a_in')
        sa_pair = tf.concat([self.s_in, self.a_in], axis=-1)
        prev_vars = len(tf.trainable_variables())

        q1_hl1 = tf.layers.dense(sa_pair, self.layer1_size, activation=tf.nn.relu, name='q1_hl1')
        q1_hl2 = tf.layers.dense(q1_hl1, self.layer2_size, activation=tf.nn.relu, name='q1_hl2')
        self.q1 = tf.layers.dense(q1_hl2, 1, activation=None, name='q1')

        q2_hl1 = tf.layers.dense(sa_pair, self.layer1_size, activation=tf.nn.relu, name='q2_hl1')
        q2_hl2 = tf.layers.dense(q2_hl1, self.layer2_size, activation=tf.nn.relu, name='q2_hl2')
        self.q2 = tf.layers.dense(q2_hl2, 1, activation=None, name='q2')

        self.q_vars = tf.trainable_variables()[prev_vars:]
        self.q = tf.minimum(self.q1, self.q2)


        #%% Critic target networks
        prev_vars = len(tf.trainable_variables())

        q1_hl1 = tf.layers.dense(sa_pair, self.layer1_size, activation=tf.nn.relu, name='q1t_hl1')
        q1_hl2 = tf.layers.dense(q1_hl1, self.layer2_size, activation=tf.nn.relu, name='q1t_hl2')
        self.q1_t = tf.layers.dense(q1_hl2, 1, activation=None, name='q1_t')

        q2_hl1 = tf.layers.dense(sa_pair, self.layer1_size, activation=tf.nn.relu, name='q2t_hl1')
        q2_hl2 = tf.layers.dense(q2_hl1, self.layer2_size, activation=tf.nn.relu, name='q2t_hl2')
        self.q2_t = tf.layers.dense(q2_hl2, 1, activation=None, name='q2_t')

        self.qt_vars = tf.trainable_variables()[prev_vars:]
        self.q_t = tf.minimum(self.q1_t, self.q2_t)

        self.v_t = self.q_t - self.alpha * self.log_prob

        #%% tf.functions (loss, optimizers)

        # Actor training
        self.pi_loss = tf.reduce_mean(self.alpha * self.log_prob - self.q)
        self.pi_optimize = tf.train.AdamOptimizer(lr_a).minimize(self.pi_loss, var_list=self.a_vars)

        # Critic training
        self.y_in = tf.placeholder(tf.float32, shape=(None,1), name='y_in')
        self.q_loss = tf.reduce_mean((self.q1 - self.y_in)**2 + (self.q2 - self.y_in)**2)
        self.q_optimize = tf.train.AdamOptimizer(lr_c).minimize(self.q_loss, var_list=self.q_vars)

        # Temperature optimization
        self.alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(self.log_prob + h))
        self.alpha_optimize = tf.train.AdamOptimizer(lr_t).minimize(self.alpha_loss)

        # Target network update
        self.q_update = tf.group([tf.assign(q_t, tau*q_t + (1-tau)*q) for q, q_t in zip(self.q_vars, self.qt_vars)])

        # Instantiate saver
        self.saver = tf.train.Saver()

    #%% Methods

    def get_action(self, s):
        ret = self.session.run([self.pi, self.log_prob], {self.s_in: s})
        return ret[0], ret[1]

    def get_value(self, s):
        ret = self.session.run(self.q, {self.s_in: s})
        return ret

    def get_target(self, s):
        ret = self.session.run(self.v_t, {self.s_in: s})
        return ret

    def get_alpha(self):
        return self.session.run(self.alpha)

    def get_det_policy(self, s):
        ret = self.session.run([self.mu, self.log_std], {self.s_in: s})
        return ret[0], ret[1]

    def get_init_value(self, s):
        a = self.session.run(self.mu, {self.s_in: s})
        return self.session.run(self.q_t, {self.s_in: s, self.a_in: a})

    def train(self, s, a, y):
#        print(len(s), s[0])
#        print(len(a), a[0])
#        print(len(y), y[0])

        self.session.run(self.q_optimize, {self.s_in: s, self.a_in: a, self.y_in: y})
        self.session.run(self.pi_optimize, {self.s_in: s})
        self.session.run(self.alpha_optimize, {self.s_in: s})

        # Get new losses
        ret0 = self.session.run(self.q_loss, {self.s_in: s, self.a_in: a, self.y_in: y})
        ret1 = self.session.run(self.pi_loss, {self.s_in: s})
        return ret0, ret1

    def update(self):
        self.session.run(self.q_update)

    def save_weigths(self, path, step):
        self.saver.save(self.session, path, global_step=step)

    def get_vis(self, s):
        ret = self.session.run([self.mu, self.pi, self.v_t, self.log_std, self.log_prob, self.noise], {self.s_in: s})
        return ret


#%% Visualizer
class MetricsTracker():
    def __init__(self, network):
        self.network = network
        self.reward_tracker = []
        self.temperature_tracker = []
        self.eplen_tracker = []
        self.mu_tracker = []
        self.pi_tracker = []
        self.log_stdev_tracker = []
        self.log_prob_tracker = []
        self.qloss_tracker = []
        self.piloss_tracker = []

        self.extras_reward_tracker = []
        self.extras_eplen_tracker = []
        self.extras_initvals_tracker = []

    def add(self, obs, r, eplen, extras={}):
        self.reward_tracker.append(r)
        self.eplen_tracker.append(eplen)
        self.temperature_tracker.append(self.network.session.run(self.network.alpha))

        if 'rew_max' in extras.keys() and 'rew_min' in extras.keys():
            self.extras_reward_tracker.append([extras['rew_max'], extras['rew_min']])

        if 'len_max' in extras.keys() and 'len_min' in extras.keys():
            self.extras_eplen_tracker.append([extras['len_max'], extras['len_min']])

        mu, pi, tg, ls, lp, no = self.network.get_vis(obs)

        self.mu_tracker.append([np.mean(mu, axis=0), np.percentile(mu, 5, axis=0), np.percentile(mu, 95, axis=0)])
        self.pi_tracker.append([np.mean(pi, axis=0), np.percentile(pi, 5, axis=0), np.percentile(pi, 95, axis=0)])
        self.log_stdev_tracker.append([np.mean(ls, axis=0), np.percentile(ls, 5, axis=0), np.percentile(ls, 95, axis=0)])
        self.log_prob_tracker.append([np.mean(lp, axis=0), np.percentile(lp, 5, axis=0), np.percentile(lp, 95, axis=0)])

        if 'init_st' in extras.keys():
            self.extras_initvals_tracker.append([np.mean(extras['init_st']),
                                                 np.min(extras['init_st']),
                                                 np.max(extras['init_st'])])
    
    def dump(self, path):
        np.save(path+'/reward_tracker.npy', np.array(self.reward_tracker))
        np.save(path+'/temperature_tracker.npy', np.array(self.temperature_tracker))
        np.save(path+'/eplen_tracker.npy', np.array(self.eplen_tracker))
        np.save(path+'/mu_tracker.npy', np.array(self.mu_tracker))
        np.save(path+'/pi_tracker.npy', np.array(self.pi_tracker))
        np.save(path+'/log_stdev_tracker.npy', np.array(self.log_stdev_tracker))
        np.save(path+'/log_prob_tracker.npy', np.array(self.log_prob_tracker))
        np.save(path+'/piloss_tracker.npy', np.array(self.piloss_tracker))
        
        np.save(path+'/extras_reward_tracker.npy', np.array(self.extras_reward_tracker))
        np.save(path+'/extras_eplen_tracker.npy', np.array(self.extras_eplen_tracker))
        np.save(path+'/extras_initvals_tracker.npy', np.array(self.extras_initvals_tracker))

    def plot(self):
######### Reward and temperature
        npr = np.array(self.reward_tracker)
        if self.extras_reward_tracker:
            x = np.arange(len(self.reward_tracker))
            npmmr = np.array(self.extras_reward_tracker)

        plt.figure(101)
        plt.clf()
        plt.plot(npr, color='tab:blue')
        if self.extras_reward_tracker:
            plt.fill_between(x, npmmr[:,0], npmmr[:,1], color='tab:blue', alpha=.1)
        plt.grid()
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.title("Test episodes rewards (avg, max, min)")
        plt.tight_layout()
        plt.draw()

######### Length and temperature
        npl = np.array(self.eplen_tracker)
        npt = np.array(self.temperature_tracker)
        if self.extras_eplen_tracker:
            x = np.arange(len(self.extras_eplen_tracker))
            npmml = np.array(self.extras_eplen_tracker)

        plt.figure(105)
        plt.clf()

#        plt.subplot(1, 2, 1)
        plt.plot(npl, color='tab:blue')
        if self.extras_eplen_tracker:
            plt.fill_between(x, npmml[:,0], npmml[:,1], color='tab:blue', alpha=.1)

        plt.grid()
        plt.ylabel("Length")
        plt.xlabel("Episode")
        plt.title("Test episode length (avg, max, min)")

        plt.tight_layout()
        plt.draw()

#        plt.subplot(1, 2, 2)
        plt.figure(106)
        plt.clf()

        plt.plot(npt, color='tab:orange')
        plt.grid()
        plt.ylabel("Temperature (alpha)")
        plt.xlabel("Episode")
        plt.title("Episode temperature")

        plt.tight_layout()
        plt.draw()

######### Internal functions/estimators
        npmu = np.array(self.mu_tracker)
        nppi = np.array(self.pi_tracker)
        npls = np.array(self.log_stdev_tracker)
        nplp = np.array(self.log_prob_tracker)
        x = np.arange(len(self.mu_tracker))

        plt.figure(102)
        plt.clf()
        for i in range(len(npmu[0,0])):
            plt.subplot(2, 2, 1)
            plt.plot(npmu[:,0,i])
            plt.fill_between(x, npmu[:,1,i], npmu[:,2,i], alpha=.1)
            plt.title('Det. Policy')

            plt.subplot(2, 2, 2)
            plt.plot(nppi[:,0,i])
            plt.fill_between(x, nppi[:,1,i], nppi[:,2,i], alpha=.1)
            plt.title('Stch. Policy')

            plt.subplot(2, 2, 3)
            plt.plot(npls[:,0,i])
            plt.fill_between(x, npls[:,1,i], npls[:,2,i], alpha=.1)
            plt.title('Log std_dev')

        plt.subplot(2, 2, 4)
        plt.plot(nplp[:,0,0])
        plt.fill_between(x, nplp[:,1,0], nplp[:,2,0], alpha=.1)
        plt.title('Log prob')

        plt.tight_layout()
        plt.draw()

######### Initial expected value
        if self.extras_initvals_tracker:
            npiev = np.array(self.extras_initvals_tracker)
            x = np.arange(len(self.extras_initvals_tracker))

            plt.figure(107)
            plt.clf()
            plt.plot(npiev[:,0])
            plt.fill_between(x, npiev[:,1], npiev[:,2], alpha=.1)
            plt.grid()
            plt.title('Initial expected value (avg, min, max)')
            plt.ylabel("Value (Q(s_0, mu(s_0))")
            plt.xlabel("Episode")
            plt.tight_layout()
            plt.draw()


        plt.pause(0.01)
