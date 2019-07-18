#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:41:50 2019

@author: haruhienomoto
"""
import matplotlib.pyplot as plt
import numpy as np


obs_num = 500
dt = 1/30
obs_time = obs_num * dt
displace_omega =  2 * np.pi/500

    
x1 = np.random.uniform(-20,20,1000)
y1 = np.random.uniform(-20,20,1000)


x2 = np.random.uniform(-20,20,1000)
y2 = np.random.uniform(-20,20,1000)


plt.figure(figsize = (6,6))  
#plt.style.use('dark_background')
plt.scatter(x1,y1,s = 3,c = 'black') 
plt.scatter(x2,y2,s = 5,c = 'black')
ax = plt.gca()                        # get current axis
ax.spines["right"].set_color("none")  
ax.spines["top"].set_color("none")  
ax.spines["left"].set_color("none")   
ax.spines["bottom"].set_color("none")   
         
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
plt.tick_params(color='white')
plt.savefig("./data/sim/f500/0001.png",dpi = 72 * 2, bbox_inches="tight", pad_inches=0.0)
#defaltだと余白が生まれてしまう., bbox_inches="tight", pad_inches=0.0で解決した
# 直交座標系 → 極座標系
radii1 = np.sqrt(x1**2 + y1**2)
theta1 = np.arctan2(y1,x1)

radii2 = np.sqrt(x2**2 + y2**2)
theta2 = np.arctan2(y2,x2)
theta_next1 = theta1
theta_next2 = theta2


for i in range(obs_num):
    plt.figure(figsize = (6,6))
    theta_next1  += displace_omega/np.sqrt(radii1)
    theta_next2  += displace_omega/np.sqrt(radii2)
    
    x1_cc = radii1 * np.cos(theta1)
    y1_cc = radii1 * np.sin(theta1)
    x2_cc = radii2 * np.cos(theta2)
    y2_cc = radii2 * np.sin(theta2)
    
    plt.figure(figsize = (6,6))
    #plt.style.use('dark_background')
    plt.scatter(x1_cc,y1_cc,s = 3,c = 'black') 
    plt.scatter(x2_cc,y2_cc,s = 5,c = 'black') 
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.tick_params(color='white')
    ax = plt.gca()                        # get current axis
    ax.spines["right"].set_color("none")  
    ax.spines["top"].set_color("none")  
    ax.spines["left"].set_color("none")   
    ax.spines["bottom"].set_color("none")   
    
    plt.savefig("./data/sim/f500/{0:04d}.png".format(i+2),dpi = 72 * 2, bbox_inches="tight", pad_inches=0.0)
    