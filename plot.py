#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:43:17 2019　
@author: 
"""

import numpy as np
import matplotlib.pyplot as plt
from random import choices
import csv

#w_iを計算する関数(重み許容範囲関数)--------------
def biw(d,Weight):
    # Weight 　= 重みの許容範囲 
    biweight = ( 1 - (d/Weight) ** 2 ) ** 2 * (abs(d/Weight) <1 )
    return (biweight)
#------------------------------------------------------------

#重み付き最小二乗法を計算-----------------------------
def culc(biweight,x,y):
    A = np.sum(biweight*x*x)
    B = np.sum(biweight*x)
    D = np.sum(biweight)
    E = np.sum(biweight*x*y)
    F = np.sum(biweight*y)
    det_matrix = A*D - B*B

    a_robust = (D*E - B*F)/det_matrix
    b_robust = (-B*E+A*F)/det_matrix
    #ロバスト法を通した、より正確なaとbをarrayにしてreturn
    culc = [a_robust,b_robust]
    return(culc)
#------------------------------------------------------------
#fitting の誤差を考慮する------------------------------------------------------- 
def ssnj_yerror(x,y):
    x_sum = np.sum(x) #B
    y_sum = np.sum(y) #F
    y_ave = y_sum/np.len(y)
    x2_sum = np.sum(x*x) #A
    delta = np.len(x)*x2_sum - x_sum*x_sum #
    
    Y_ave =np.array([y_ave]*len(x))
    Y_var = y -Y_ave   # array
    σ_obs = Y_var*Y_var
    σ_y = np.sqrt(np.sum(σ_obs)/(len(x)-2)) 

    σ_a =σ_y *np.sqrt(len(x)/delta) 
    σ_b =σ_y *np.sqrt(x2_sum/delta) 
    array = [σ_a,σ_b]
    
    return(array)
#------------------------------------------------------------ 

#複数回計算する関数(robust法本体)-------------------
def robust(x,y,Weight_func):
    res1=np.polyfit(x, y, 1)
    d = []
    d =  y  - (res1[0] * x +res1[1])
    
    for i in Weight_func:
        #重み許容範囲関数を呼び出し
        biweight = biw(d,i) 
        #重み付き最小二乗法を呼び出し
        robust_result = culc(biweight,x,y)
        d = y - (robust_result[0]*x + robust_result[1])
    return(robust_result)
#------------------------------------------------------------ 

#Boot strap法本体--------------------------------------
def bootstrap(x,y,repeats):
    Weight_btp = [0.05]*10
    search_ideala = [] #aを集める配列を準備
    search_idealb = [] #bを集める配列を準備

    for i in range(repeats):
        index = np.arange(0,len(x),1)
        rand_index= choices(index, k=130) #indexでRandom Sampling(重複許す)
        x_resample = []
        y_resample = []
        for i in rand_index:
            x_resample.append(x[i])
            y_resample.append(y[i])
            arr_rex = np.array(x_resample)
            arr_rey = np.array(y_resample)

        #ロバスト法を計算
        result_btp = robust(arr_rex,arr_rey,Weight_btp)
        search_ideala.append(result_btp[0])
        search_idealb.append(result_btp[1])
        
    return(np.mean(search_ideala) ,np.mean(search_idealb))
    
#------------------------------------------------------------ 


def plot(u,v,n,k,arr,time):

    NX=n
    NY=n
    xvmin=-arr.shape[0]/2.
    xvmax=arr.shape[0]/2
    yvmin=-arr.shape[1]/2
    yvmax=arr.shape[1]/2
    xcmin = xvmin - (xvmax - xvmin)/(2.*NX)
    xcmax = xvmax + (xvmax - xvmin)/(2.*NX)
    ycmin = yvmin - (yvmax - yvmin)/(2.*NY)
    ycmax = yvmax + (yvmax - yvmin)/(2.*NY)

    ## カラープロットのグリッド
    xc  =np.linspace(xcmin,xcmax,NX+1)
    yc = np.linspace(ycmin,ycmax,NY+1)
    XC, YC = np.meshgrid(xc,yc)

    ## ベクトルのグリッド
    xv = np.linspace(xvmin,xvmax,NX)
    yv = np.linspace(yvmin,yvmax,NY)
    XV,YV = np.meshgrid(xv,yv)

    rho =np.sqrt(u**2 + v**2)
    rho_t = rho[::-1,:]
    plt.figure(figsize = (6,6)) # barあるときは8,6
    #plt.pcolormesh(XC,YC,rho_t,alpha = 0.6 ,vmin= 0, vmax = 15)#14
    #plt.colorbar()
    u_t = u[::-1,:]
    v_t = v[::-1,:]
    plt.quiver(XV,YV,u_t,v_t)
    plt.plot(0,0,marker = "+")
    #plt.scatter(XC,YC)
    #plt.scatter(XV,YV)
    plt.xlim([xcmin,xcmax])
    plt.ylim([ycmin,ycmax])
    plt.tick_params(color='white')
    plt.title('t = %f [sec]' % (time))
    #plt.show()
    #plt.savefig("./data/data/070306/result/re{0:04d}.png".format(k+1))
    
    

def plotr(u_list,v_list,R_list,k,Theta_list):
    time = k/30.
    v_r = u_list* np.cos(Theta_list) + v_list* np.sin(Theta_list)
    v_th =  - 1. * u_list * np.sin(Theta_list) + v_list * np.cos(Theta_list)
    vec = np.sqrt(v_th **2)
    vec_reshape = np.reshape(vec, (vec.shape[0]**2,))
    R_reshape = np.reshape(R_list,(R_list.shape[0]**2,))
    
    #変位0のデータだけ消去
    kumi = np.stack((vec_reshape,R_reshape),axis = -1)
    vec_remove0 = []
    R_remove0 = []
    for i in range(kumi.shape[0]):
        if kumi[i,0] > 0 and kumi[i,1] >1 and kumi[i,0] < 20:
            vec_remove0.append(kumi[i,0])
            R_remove0.append(kumi[i,1])

    vec_ar = np.array([vec_remove0])
    
    #fitting
    z = 1/np.sqrt(R_remove0)  #convert to np.array
    
    #Weight= [0.05,0.05,0.05,0.05]
    
    result = bootstrap(z, vec_ar.T,10)
    z_model = np.linspace(0.04,0.16,len(z))
    y_fitting =  result[0] * z_model + result[1] 
    ####================================--=======
    plt.figure()
    plt.scatter(z,vec_ar,s = 1,c = 'red')
    plt.scatter(z_model,y_fitting,s = 1,c = 'blue')
    plt.title('t = %f [sec]' % (time))
    #plt.plot(z_model,y_fitting)
    #plt.savefig("./data/data/070306/result_z/radius{0:04d}.png".format(k+1))
    ####================================--=======
    
    R_model = np.linspace(50,500,len(y_fitting))
    y_fitR = result[0] / np.sqrt(R_model) + result[1] 

    #plot
    plt.figure(figsize = (8,8))
    plt.scatter(R_remove0,vec_remove0,c = 'darkcyan',s = 5)
    plt.plot(R_model,y_fitR,c = 'red')
    
    plt.title('t = %f [sec]' % (time))
    plt.xlabel('Radius  [pixel]')
    plt.ylabel('Deplacement  [pixel]')
    plt.ylim([0,10])
    #plt.savefig("./data/data/070306/result_fit/radius{0:04d}.png".format(k+1))

    return result[0],result[1]
    
    
    
    
