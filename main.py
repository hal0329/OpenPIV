#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:29:36 2019
@author: haruhienomoto
"""

#解像度  -> 1080 *720 なら仕方ないのではという結論
#相互相関係数 高すぎるがそういうものらしい

import numpy as np
#import matplotlib.pyplot as plt
import cv2
#import forwardP
import cc
import plot
import postP
import csv
#=============================================================================
obs_num =  499                     # 解析する画像枚数len(data image)- 1 
#total_num = 50
dt = 1/30                            #[sec] Measurement interval
obsavation_time = dt*obs_num
dev_num = 17
center = int(np.floor(dev_num/2))
#=============================================================================


a = 1j
x_list = np.arange(-(720/dev_num)*np.floor(dev_num/2),(720/dev_num)*(np.floor(dev_num/2)+1),(720/dev_num))
y_list = np.arange((720/dev_num)*np.floor(dev_num/2),-(720/dev_num)*(np.floor(dev_num/2)+1),-(720/dev_num))
y_list_img = a* y_list
X,Y = np.meshgrid(x_list,y_list_img)
Z_list  = X + Y
R_list = np.absolute(Z_list)
Theta_list = np.rad2deg(np.angle(Z_list))

#forward process
#img_ave_sp = forwardP.start(total_num)
#print('forward process was succeseeded.')
u_kaiseki = np.zeros((dev_num ,dev_num ))
v_kaiseki = np.zeros((dev_num ,dev_num ))

for k in range(obs_num):
    print('step',k)
    #forward processing
    img1_raw = cv2.imread("./data/sim/f500/{0:04d}.png".format(k+1),0)
    img2_raw = cv2.imread("./data/sim/f500/{0:04d}.png".format(k+3),0)
    #background reduction
    img1 = img1_raw #- img_ave_sp
    img2 = img2_raw #- img_ave_spR    

    #image enhancement
    
    #separate into dev_num**2 pieces
    img_arr = cc.separate(img1,dev_num,dev_num)
    img_next_arr = cc.separate(img2,dev_num,dev_num)
    pixel = img_arr.shape[1]
    
    Cxy_li = [] #毎回listで定義し,np.arrayに変換して出力。 #deplacement list
    u_list = np.zeros(dev_num**2,)
    
    v_list = np.zeros(dev_num**2,)
    
    for i in range(dev_num**2):
        img = img_arr[i,:,:]
        img_next = img_next_arr[i,:,:]
        img_float = np.float64(img)
        img_float2 = np.float64(img_next)
        
        
        img_li = []
        img_li.append(img_float)
        img_li.append(img_float2)

        d, Cxy_s,ans = cc.myphaseCorrelate(img_float2,img_float)
        u_list[i] = d[0]  #order #*framerate ?
        v_list[i] = d[1] #*framerate ?
        Cxy_li.append(Cxy_s)
        out_Cxy = np.array(Cxy_li)
        if ans ==1:
            print(i//dev_num,i - i//dev_num*dev_num)
    
    
    u_list_reshape = np.reshape(u_list, (dev_num, dev_num))
    v_list_reshape = np.reshape(v_list, (dev_num, dev_num)) 
    #post prodessing #周辺ベクトルで補完
    u_list_ip = postP.vecInterpolation(u_list_reshape)
    v_list_ip = postP.vecInterpolation(v_list_reshape)

    u_kaiseki_kousin = np.dstack([u_kaiseki, u_list_ip])
    u_kaiseki = u_kaiseki_kousin
    v_kaiseki_kousin = np.dstack([v_kaiseki, v_list_ip])
    v_kaiseki = v_kaiseki_kousin

#時系列で補完(過去と未来から現在を内挿) 今は前後5枚ずつを内挿
u_time = postP.ContinuityofTime(u_kaiseki)
print('kaiseki_uok')
v_time = postP.ContinuityofTime(v_kaiseki)
print('kaiseki_vok')

v_r_kaiseki = np.zeros((dev_num ,dev_num ))
v_th_kaiseki = np.zeros((dev_num ,dev_num ))
"""
with open('./data/kaiseki/070306_dev17.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['trial', 'j','u','v','v_r','v_th','vec','radius'])
    for i in range(obs_num):
        if i > 1:
            u_i = u_time[:,:,i]
            u_i_reshape = np.reshape(u_i, (u_i.shape[0]**2,))
            v_i = v_time[:,:,i]
            v_i_reshape = np.reshape(v_i, (v_i.shape[0]**2,))
            vec = np.sqrt(u_i **2 + v_i  **2)
            vec_reshape = np.reshape(vec, (vec.shape[0]**2,))
            R_reshape = np.reshape(R_list,(R_list.shape[0]**2,))
            
            v_r = u_time[:,:,i] * np.cos(Theta_list) + v_time[:,:,i] * np.sin(Theta_list)
            v_r_reshape = np.reshape(v_r, (v_r.shape[0]**2,))
            v_th =  - 1. * u_time[:,:,i] * np.sin(Theta_list) + v_time[:,:,i] * np.cos(Theta_list)
            v_th_reshape = np.reshape(v_th, (v_th.shape[0]**2,))
            
            v_r_kousin = np.dstack([v_r_kaiseki,v_r])
            v_r_kaiseki = v_r_kousin
            v_th_kousin = np.dstack([v_th_kaiseki,v_th])
            v_th_kaiseki = v_th_kousin
            
            
            for j in range(R_reshape.shape[0]):
                writer.writerow([i-1,j,u_i_reshape[j],v_i_reshape[j],v_r_reshape[j],v_th_reshape[j],vec_reshape[j],R_reshape[j]])
"""
    
"""  
X_r = np.mean(v_r_kaiseki,axis = 2) - np.std(v_r_kaiseki,axis = 2)
X_th = np.mean(v_th_kaiseki,axis = 2) - np.std(v_th_kaiseki,axis = 2)
print(np.round(X_r,3))
print(np.round(X_th,3))

plt.figure()
plt.scatter(R_reshape,X_r,s  = 1)
plt.show()

plt.figure()
plt.scatter(R_reshape,X_th,s  = 1)
plt.show()
"""

#a,bを探す
a_list = []
b_list = []

for i in range(obs_num):
    if i>1:
        print('plotstep@',i)
        #plot.plot(u_time[:,:,i],v_time[:,:,i],dev_num,i,img1,i *dt)
        a,b = plot.plotr(u_time[:,:,i],v_time[:,:,i],R_list,i,Theta_list)
        a_list.append(a)
        b_list.append(b)
       
a_ave_list = []
b_ave_list = []

for k in range(int(obs_num/30)):
    a_ave_30 = np.mean(a_list[k *30:(k+1)*30])
    b_ave_30 = np.mean(b_list[k *30:(k+1)*30])
    a_ave_list.append(a_ave_30)
    b_ave_list.append(b_ave_30)

"""
with open('./data/kaiseki/070306_ab.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['trial', 'a','b'])
    for i in range(len(a_ave_list)):
        writer.writerow([i+1,a_ave_list[i],b_ave_list[i]])
    #pcolormesh y軸方向に反転? imshowとの違い
    #img = np.delete(img, slice(2573,2559,1), 1) #slice できない...始めから正方行列にする!
    
"""