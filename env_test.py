#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from quad_env import QuadEnv


env = QuadEnv()

for i in range(3):
    obs = env.reset()
    
    while True:
        action = np.random.uniform(-1,1,4)
        
        env.render()
        time.sleep(0.0033)
        
        observation, reward, done, info = env.step(action)
        
        if done: break
    
