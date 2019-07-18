#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:00:12 2019
@author: haruhienomoto
"""
import numpy as np
import matplotlib.pyplot as plt

def comparison(sr_li,center):
    
    C1 = 0. #考慮の余地あり (PIVハンドブック[森元出版]p.111 を参照)
    C2 = 1.5#考慮の余地あり
    res_surr = np.reshape(sr_li, (1,sr_li.shape[0]*sr_li.shape[1]))
    #print(np.std(res_surr))
    if abs(center - np.mean(res_surr)) <= (C1 + C2 * np.std(res_surr)):
        return 0
    else:
        return 1
    
def lagrangeInterpolation(surr_li,center):
    weight = 1/np.sqrt(2)
    cross = surr_li[0,0] + surr_li[0,2] + surr_li[2,0] + surr_li[2,2]
    plus = surr_li[0,1] + surr_li[1,0] + surr_li[1,2] + surr_li[2,1]
    u_new = 1/4/(1 + weight)*plus +weight/4/(1 + weight)*cross
    return u_new

def vecInterpolation(u_list):
    n = u_list.shape[0]
    u_list_new = u_list
    u_check = np.zeros((u_list.shape[0],u_list.shape[0]))    
    
    for i in range(n):
        weight = 1/np.sqrt(2)
        for j in range(n):
            if (i == 0 and j == 0):
                #area 1
                u_sur = u_list[0:2,0:2]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    plus =  u_sur[1,0] + u_sur[0,1]
                    cross = u_sur[1,1]
                    u_list_new[i,j] = 1/2/(1 + weight)*plus +weight/1/(1 + weight)*cross
                else:
                    pass
                
            elif (i == 0 and j == n-1):
                #area 3
                u_sur = u_list[0:2,n-2:n]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    plus =  u_sur[0,0] + u_sur[1,1]
                    cross = u_sur[1,0] 
                    u_list_new[i,j] = 1/2/(1 + weight)*plus +weight/1/(1 + weight)*cross
                else:
                    pass
                    
            elif (i == n-1 and j == 0):
                #area 7
                u_sur = u_list[n-2:n,0:2]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    plus =  u_sur[0,0] + u_sur[1,1]
                    cross = u_sur[0,1]
                    u_list_new[i,j] = 1/2/(1 + weight)*plus +weight/1/(1 + weight)*cross
                else:
                    pass
                   
                        
            elif (i == n-1 and j == n-1):
                #area 9
                u_sur = u_list[n-2:n,n-2:n]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    plus =  u_sur[1,0] + u_sur[0,1]
                    cross = u_sur[0,0]
                    u_list_new[i,j] = 1/2/(1 + weight)*plus +weight/1/(1 + weight)*cross
                else:
                    pass
                   
            elif (i != 0 and i != n-1 and j == 0):
                #area 4
                u_sur = u_list[i-1:i+2,0:2]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    plus =  u_sur[0,0] + u_sur[1,1] + u_sur[2,0]
                    cross = u_sur[0,1] + u_sur[2,1]
                    u_list_new[i,j] = 1/3/(1 + weight)*plus +weight/2/(1 + weight)*cross
                else:
                    pass
   
            elif (i == 0 and j != 0 and j != n-1):
                #area 2
                u_sur = u_list[0:2,j-1:j+2]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    plus =  u_sur[0,0] + u_sur[0,2] + u_sur[1,1]
                    cross = u_sur[1,0] + u_sur[1,2]
                    u_list_new[i,j] = 1/3/(1 + weight)*plus +weight/2/(1 + weight)*cross
                else:
                    pass
                           
            elif (i != 0 and i != n-1 and j == n-1):
                #area 6
                u_sur = u_list[i-1:i+2,n-2:n]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    plus =  u_sur[0,1] + u_sur[1,0] + u_sur[2,1]
                    cross = u_sur[0,0] + u_sur[2,0]
                    u_list_new[i,j] = 1/3/(1 + weight)*plus +weight/2/(1 + weight)*cross
                else:
                    pass
   
            elif (i == n-1 and j != 0 and j != n-1):
                #area 8
                u_sur = u_list[n-2:n,j-1:j+2]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    plus =  u_sur[1,0] + u_sur[0,1] + u_sur[1,2]
                    cross = u_sur[0,0] ++ u_sur[0,2]
                    u_list_new[i,j] = 1/3/(1 + weight)*plus +weight/2/(1 + weight)*cross
                else:
                    pass
   
            else:
                #area 5
                u_sur = u_list[i-1:i+2,j-1:j+2]
                test = comparison(u_sur,u_list[i,j])
                if test == 1:
                    u_check[i,j] = 1
                    u_list_new[i,j] = lagrangeInterpolation(u_sur,u_list[i,j])
                else:
                    pass
  
    #print(u_check)
    return u_list_new

def continuousInterpolation(u_now,u_before):
    for i in range(u_now.shape[0]):
        for j in range(u_now.shape[1]):
            if np.abs(u_now[i,j] - u_before[i,j]) > 1.5:
                u_now[i,j] = u_before[i,j]
                
    return u_now


def estmationOfVerticity(pixel,u_list,v_list):
    n = u_list.shape[0]
    omega_list = np.zeros_like((u_list))
    
    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0):
                #area 1
                delvdelx =  (u_list[i+1,j] - u_list[i,j])/pixel
                deludely = (v_list[i,j+1] - v_list[i,j])/pixel
                omega_list[i,j] = delvdelx - deludely

            elif (i == 0 and j == n-1):
                #area 3
                delvdelx = (u_list[i+1,j] - u_list[i,j])/pixel
                deludely = (v_list[i,j] - v_list[i,j-1])/pixel
                omega_list[i,j] = delvdelx - deludely
                    
            elif (i == n-1 and j == 0):
                #area 7
                delvdelx  =  (u_list[i,j] - u_list[i-1,j])/pixel
                deludely = (v_list[i,j+1] - v_list[i,j])//pixel
                omega_list[i,j] = delvdelx - deludely
                        
            elif (i == n-1 and j == n-1):
                #area 9
                delvdelx = (u_list[i,j] - u_list[i-1,j])/pixel
                deludely = (v_list[i,j] - v_list[i,j-1])/pixel
                omega_list[i,j] = delvdelx - deludely
                
            elif (i != 0 and i != n-1 and j == 0):
                #area 4
                delvdelx = (u_list[i+1,j] - u_list[i-1,j])/2/pixel
                deludely = (v_list[i,j+1] - v_list[i,j])/pixel
                omega_list[i,j] = delvdelx - deludely

            elif (i == 0 and j != 0 and j != n-1):
                #area 2
                delvdelx =  (u_list[i+1,j] - u_list[i,j])/pixel
                deludely = (v_list[i,j+1] - v_list[i,j-1])/2/pixel
                omega_list[i,j] = delvdelx - deludely
                
            elif (i != 0 and i != n-1 and j == n-1):
                #area 6
                delvdelx = (u_list[i+1,j] - u_list[i-1,j])/2/pixel
                deludely = (v_list[i,j] - v_list[i,j-1])/pixel
                omega_list[i,j] =delvdelx - deludely

            elif (i == n-1 and j != 0 and j != n-1):
                #area 8
                delvdelx =  (u_list[i,j] - u_list[i-1,j])/pixel
                deludely = (v_list[i,j+1] - v_list[i,j-1])/2/pixel
                omega_list[i,j] = delvdelx - deludely
                
            else:
                if (i == 1 or i == n-2 or j == 1 or j == n-2):
                    delvdelx = (u_list[i+1,j] - u_list[i-1,j])/2/pixel
                    deludely = (v_list[i,j+1] - v_list[i,j-1])/2/pixel
                    omega_list[i,j] = delvdelx - deludely
                    
                else:
                    delvdelx = (-v_list[i+2,j] + 4 * v_list[i+1,j] - 4 * v_list[i-1,j] + v_list[i-2,j])/4/pixel
                    deludely = (-u_list[i,j+2] + 4 * u_list[i,j+1] - 4 * u_list[i,j-1] +u_list[i,j-2])/4/pixel
                    omega_list[i,j] = delvdelx - deludely
                    

    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    """
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(omega_list[::1,:], interpolation='none',vmin = -4,vmax = 4)
    fig.colorbar(im)
    plt.show()
    """
    return omega_list

def ContinuityofTime(u_arr):
    obs_num_p1 = u_arr.shape[2]
    u_return= np.zeros((u_arr.shape[0],u_arr.shape[0]))
    
    for i in range(obs_num_p1):
        if i>5:
            u_isec = u_arr[:,:,i]
            #print(u_arr[:,:,i-5:i+5].shape) 最後でもこのまま行けるのは割と謎
            u_surr_mean = np.mean(u_arr[:,:,i-5:i+5],axis = 2)
            u_surr_var = np.std(u_arr[:,:,i-5:i+5],axis = 2)
            u_surr_var_not0 =np.where(u_surr_var == 0, 0.001, u_surr_var)


            u_test = (u_isec - u_surr_mean)/u_surr_var_not0
            
            #print(np.round(u_test),4)
            u_new = np.where(u_test >= 2, u_surr_mean, u_isec)
            #print(u_new)
            u_return_kousin = np.dstack([u_return, u_new])
            u_return = u_return_kousin
            
        else:
            u_isecelse = u_arr[:,:,i]
            u_return_kousin = np.dstack([u_return, u_isecelse])
            u_return = u_return_kousin
    
    return u_return         
