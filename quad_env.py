#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import sys

# https://github.com/abhijitmajumdar/Quadcopter_simulator
from quadcopter import Quadcopter, Propeller
from gui import GUI


#%% Reward functions
def reward_function_0(s, a, sp, outside):
    ''' Try to keep the quad in hover'''
    if outside:
        rx = -1000
    else:
        rx = 0

    rv = -1.0 * np.sum(np.sqrt(sp[3:6]**2))
    rth = -1.0 * np.sum(np.sqrt(sp[6:9]**2))
    rw = -1.0 * np.sum(np.sqrt(sp[9:12]**2))

    return rx+rv+rth+rw


def reward_function_origin(s, a, sp, outside):
    ''' Try to keep the quad hovering in [0,0]'''
    if sp[2] <= 0:
        rx = -1000
    else:
        rx = -1.0 * np.sum(np.sqrt(sp[0:2]**2))
        
    rv = -1.0 * np.sum(np.sqrt(sp[3:6]**2))
    rth = -1.0 * np.sum(np.sqrt(sp[6:9]**2))
    rw = -1.0 * np.sum(np.sqrt(sp[9:12]**2))

    return rx+rv+rth+rw


#%% Spawn randomizers
def spawn_anywhere():
    xy = np.random.uniform(-1,1,2)
    z = np.random.uniform(1,4,1)
    vel = np.random.uniform(-2,2,3)
    th = np.random.uniform(-0.3,0.3,3)
    w = np.array([0,0,0])

    return np.hstack([xy, z, vel, th, w])


def spawn_aggressive():
    xy = np.random.uniform(-0.5,0.5,2)
    z = np.random.uniform(2,3,1)
    vel = np.random.uniform(-1,1,3)
    th = np.random.uniform(-0.1,0.1,3)
    w = np.array([0,0,0])

    # Aggressive velocity XOR attitude
    if np.random.uniform() > 0.5:
        vel = np.random.uniform(-4,4,3)
    else:
        th = np.random.uniform(-1,1,3)

    return np.hstack([xy, z, vel, th, w])

def spawn_even_more_aggressive():
    xy = np.random.uniform(-0.5,0.5,2)
    z = np.random.uniform(3,4,1)
    vel = np.random.uniform(-3,3,3)
    th = np.random.uniform(-1,1,3)
    w = np.array([0,0,0])

    return np.hstack([xy, z, vel, th, w])

def spawn_randomizer():
    roll = np.random.uniform()
    if roll > 0.4:
        return spawn_anywhere()
    elif roll > 0.1:
        return spawn_aggressive()
    else:
        return spawn_even_more_aggressive()

#%% Environment class
class QuadEnv():
    def __init__(self):

        # initial position and orientation, length of arm, center radius, propeller size and weight
        self.quad_properties = {'q1':{'position':[0,0,0],'orientation':[0,0,0],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2}}
        self.quad = Quadcopter(self.quad_properties.copy())

        self.quad.quads['q1']['state'] = spawn_randomizer()   # Set spawning properties here

        self.dt = 0.02
        self.count = 0
        self.gui_object = None

    def allocate_control(self, actions):

        actions[0] = np.clip(actions[0], 0, 1)
        actions[1:] = np.clip(actions[1:], -1, 1)

        delta = np.array([[ 1,  1,  1, -1],
                          [ 1, -1,  1,  1],
                          [ 1, -1, -1, -1],
                          [ 1,  1, -1,  1]])

        MIN_RPM = 3000
        MAX_RPM = 9000
        return np.clip((delta@actions)*MAX_RPM, MIN_RPM, MAX_RPM)

    def reset(self):
        self.__init__()
        return self.quad.get_state('q1')


    def step(self, action):

        self.count += 1
        s = self.quad.get_state('q1')
        a = self.allocate_control(action)
        self.quad.set_motor_speeds('q1', a)
        self.quad.update(self.dt)

        isOutside = self.let_it_cross()              # Set border crossing behavior here
        obs = self.quad.get_state('q1')
        rew = reward_function_origin(s, a, obs, isOutside)   # Set reward function here
        done = bool(self.count >= 1000)

        return obs, rew, done, {'steps':self.count}

    def render(self):

        if self.gui_object is None: self.gui_object = GUI(self.quad_properties)

        self.gui_object.quads['q1']['position'] = self.quad.get_position('q1')
        self.gui_object.quads['q1']['orientation'] = self.quad.get_orientation('q1')
        self.gui_object.update()
        
#%% Lock drone in testing area
    def lock_inside(self):
        sp = self.quad.get_state('q1')
        isOutside = False
        if np.abs(sp[0]) >= 2.0:
            self.quad.quads['q1']['state'][0] = np.sign(sp[0])*2.0
            isOutside = True
        if np.abs(sp[1]) >= 2.0:
            self.quad.quads['q1']['state'][1] = np.sign(sp[1])*2.0
            isOutside = True
        if sp[2] <= 0.0:
            self.quad.quads['q1']['state'][2] = 0.0
            isOutside = True
        if sp[2] >= 5.0:
            self.quad.quads['q1']['state'][2] = 5.0
            isOutside = True
        
        return isOutside
    
    def let_it_cross(self):
        sp = self.quad.get_state('q1')
        if abs(sp[0])>=2.0 or abs(sp[1])>=2.0 or sp[2]<=0.0 or sp[2]>=5.0:
            return True
        else:
            return False
        
    def wrap_around(self):
        sp = self.quad.get_state('q1')
        isOutside = False
        if np.abs(sp[0]) >= 2.0:
            self.quad.quads['q1']['state'][0] = np.sign(sp[0])*-2.0
            isOutside = True
        if np.abs(sp[1]) >= 2.0:
            self.quad.quads['q1']['state'][1] = np.sign(sp[1])*-2.0
            isOutside = True
        if sp[2] <= 0.0:
            self.quad.quads['q1']['state'][2] = 0.0
            isOutside = True
        if sp[2] >= 5.0:
            self.quad.quads['q1']['state'][2] = 5.0
            isOutside = True
        
        return isOutside
