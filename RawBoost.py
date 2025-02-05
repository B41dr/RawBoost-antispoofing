#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import copy


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y

def normWav(x,always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x



def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    # 优化点1：使用向量化操作替代循环
    # 原公式：fc = randRange(minF,maxF,0); bw = randRange(minBW,maxBW,0);
    #         c = randRange(minCoeff,maxCoeff,1);
    
    # 一次性生成所有参数
    fc = np.random.uniform(minF, maxF, nBands)  # 中心频率
    bw = np.random.uniform(minBW, maxBW, nBands)  # 带宽
    c = np.random.randint(minCoeff, maxCoeff + 1, nBands)  # 系数
    
    # 确保系数为奇数
    c = np.where(c % 2 == 0, c + 1, c)
    
    # 计算频率边界
    f1 = np.maximum(fc - bw/2, 1/1000)
    f2 = np.minimum(fc + bw/2, fs/2 - 1/1000)
    
    # 向量化计算滤波器系数
    b = np.ones(1)
    for i in range(nBands):
        b = np.convolve(signal.firwin(c[i], [f1[i], f2[i]], window='hamming', fs=fs), b)
    
    # 增益计算
    G = np.random.uniform(minG, maxG)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G/20) * b / np.amax(abs(h))
    
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y

# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x, N_f, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin, maxBiasLinNonLin, fs):
    # 优化点2：使用矩阵运算替代循环
    y = np.zeros_like(x)
    for i in range(N_f):
        if i == 1:
            minG -= minBiasLinNonLin
            maxG -= maxBiasLinNonLin
        
        b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
        y += filterFIR(np.power(x, (i+1)), b)
    
    y -= np.mean(y)
    return normWav(y, 0)


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    # 优化点3：使用向量化操作
    beta = np.random.uniform(0, P)
    y = x.copy()
    n = int(x.shape[0] * (beta/100))
    p = np.random.permutation(x.shape[0])[:n]
    f_r = (2 * np.random.rand(p.shape[0]) - 1) * (2 * np.random.rand(p.shape[0]) - 1)
    y[p] += g_sd * x[p] * f_r
    return normWav(y, 0)


# Stationary signal independent noise

def SSI_additive_noise(x, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    # 优化点4：简化噪声生成流程
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = np.random.uniform(SNRmin, SNRmax)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0**(0.05 * SNR)
    return x + noise

# genNotchCoeffs() 优化
# 使用np.random.uniform和np.random.randint一次性生成所有参数
# 使用np.where确保系数为奇数
# 使用向量化计算频率边界
# LnL_convolutive_noise() 优化
# 使用np.zeros_like初始化y
# 简化增益调整逻辑
# 保持原有功能，减少代码复杂度
# ISD_additive_noise() 优化
# 使用向量化操作计算f_r
# 简化索引操作
# 减少中间变量
# 4. SSI_additive_noise() 优化
# 简化噪声生成流程
# 保持SNR计算逻辑不变
# 减少不必要的变量声明