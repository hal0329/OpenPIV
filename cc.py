#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:03:45 2019

@author: haruhienomoto
"""

import numpy as np
from scipy import fftpack
#import matplotlib.pyplot as plt

def separate(img,v_sp,h_sp):
    #crop
    h, w = img.shape[:2]
    crop_img = img[:h // v_sp * v_sp, :w // h_sp * h_sp]
    #print('{} -> {}'.format(img.shape, crop_img.shape))
 
    #splits
    out_imgs = []
    for h_img in np.vsplit(crop_img, v_sp): #vertical direction
        for v_img in np.hsplit(h_img,h_sp):  #horizontal direction<->
            out_imgs.append(v_img)
    out_imgs_np = np.array(out_imgs)
    #print(out_imgs.shape)
    """
    fig, ax_list = plt.subplots(v_sp, h_sp, figsize=(5, 5))
    for sub_img, ax in zip(out_imgs_np, ax_list.ravel()):
        ax.imshow(x) #なぜかxでうまく行っていたがこれはsub_imgでは?
        ax.set_axis_off()
    plt.show()
    """
    
    return out_imgs_np

def myphaseCorrelate(img2,img1): #1:t[s] #2:t+dt[s]
    ans = 0
    nx0_img = float(img1.shape[1])
    ny0_img = float(img1.shape[0])
    pixel = nx0_img*ny0_img
    f1 = fftpack.fft2(img1)   
    f2 = fftpack.fft2(img2)
    
    Pxy = np.multiply(np.conj(f1),f2)/pixel
    Cxy = fftpack.ifft2(Pxy)
    
    Pxx = np.multiply(np.conj(f1),f1)/pixel
    Cxx = np.real(fftpack.ifft2(Pxx))
    Pyy = np.multiply (np.conj(f2),f2)/pixel
    Cyy = np.real(fftpack.ifft2(Pyy))
    
    Cxy_s = fftpack.fftshift(np.real(Cxy))/np.sqrt(Cxx[0,0]*Cyy[0,0])
    #Cxy_s = np.real(Cxy)   
    dum = np.unravel_index(np.argmax(Cxy_s), Cxy_s.shape)
    #index 配列を一緒にsort? 配列ごとにsortして最後に全体でsort
    
    dx = dum[1]-np.floor(nx0_img/2.)
    dy = -1.*(dum[0]-np.floor(ny0_img/2.))
    d = [dx,dy]
    
    
    if np.sum(Cxy_s) > 0.99999 * Cxy_s.shape[0] * Cxy_s.shape[1]:
        d = [0,0] #simulation画像の余白への操作
        ans = 1
        
    if np.sum(Cxy_s) < 0.0001 * Cxy_s.shape[0] * Cxy_s.shape[1]:
        d = [0,0] #simulation画像の余白への操作
        ans = 1
        
    

    return d, Cxy_s,ans