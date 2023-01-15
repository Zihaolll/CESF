#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:36:50 2021

@author: zh
"""



'''
def _leaves_grad(input_sample, leaves_interval, leaves_prob, grad_fnc=_indicatior_grad)

input:样本，叶子路径矩阵，叶子路径对应的分类概率，计算导数的方法
output:基于分类标签的路径梯度
'''




import numpy as np
import math
import sys
#%%
def _indicator_function_leaf_interval(input_sample, leaf_interval):#输入的叶子区间仅为单一标签的
    # leaf_interval = np.array(leaf_interval)
    curr_leaf_interval = leaf_interval.copy()
    diff_0 = curr_leaf_interval[:,:,0] - input_sample#左边界-样本点
    diff_1 = curr_leaf_interval[:,:,1] - input_sample#右边界-样本点
    mask_in = np.array((diff_0 * diff_1) <= 0, dtype = int)#判断样本点在是否区间内的mask，示性矩阵
    mask_left = np.array(diff_0 > 0, dtype = int)#判断样本点是否在区间左侧的mask
    mask_right = np.array(diff_1 < 0, dtype = int)#判断样本点是否在区间右侧的mask
    return mask_in, mask_left, mask_right, diff_0, diff_1

#%%
def _indicatior_grad(input_sample, leaf_interval):
    mask_in, mask_left, mask_right, diff_0, diff_1 =_indicator_function_leaf_interval(input_sample, leaf_interval)
    mask1 = np.array(mask_in == 0,dtype = int)#区间内部为0的mask
    mask2 = np.ones(mask_right.shape)
    mask2[np.array(mask_right,dtype=bool)] = -1#标记样本点在区间右侧时梯度为负的mask
    grad = np.minimum(np.abs(diff_0),np.abs(diff_1)) + 1e-5

    grad = 1/grad * mask1 * mask2

    return grad, mask_in #梯度矩阵以及示性矩阵

def _indicatior_grad_cos(input_sample, leaf_interval, data_range = 1):#当前winedata范围为1
    mask_in, mask_left, mask_right, diff_0, diff_1 =_indicator_function_leaf_interval(input_sample, leaf_interval)
    mask1 = np.array(mask_in == 0,dtype = int)#区间内部为0的mask
    mask2 = np.ones(mask_right.shape)
    mask2[np.array(mask_right,dtype=bool)] = -1#标记样本点在区间右侧时梯度为负的mask
    grad = np.minimum(np.abs(diff_0),np.abs(diff_1))
    grad = (grad / data_range) * (math.pi / 4) + math.pi / 4
    grad = np.cos(grad) * mask1 * mask2
    return  grad, mask_in #梯度矩阵以及示性矩阵

def _indicatior_grad_dis(input_sample, leaf_interval, data_range = 1):#当前winedata范围为1
    mask_in, mask_left, mask_right, diff_0, diff_1 =_indicator_function_leaf_interval(input_sample, leaf_interval)
    mask1 = np.array(mask_in == 0,dtype = int)#区间内部为0的mask
    mask2 = np.ones(mask_right.shape)
    mask2[np.array(mask_right,dtype=bool)] = -1#标记样本点在区间右侧时梯度为负的mask
    grad = np.minimum(np.abs(diff_0),np.abs(diff_1))
    grad = grad / data_range
    grad = grad * mask1 * mask2
    return  grad, mask_in #梯度矩阵以及示性矩阵
#%%
def _leaves_grad(input_sample, leaves_interval, leaves_prob, grad_fnc=_indicatior_grad):
    indicator_grad ,indicator= grad_fnc(input_sample, leaves_interval)
    leaves_grad_matrix = np.ones(indicator.shape)
    leaves_grad_class = []
    for i in range(indicator.shape[1]):
        leaves_grad_matrix[:,i] = indicator_grad[:,i] * np.prod(indicator[:,0:i],axis=1) * np.prod(indicator[:,i+1:],axis=1)
        #控制当前样本与叶子区域除待计算维度外在同超平面内
    for i in range(leaves_prob.shape[2]):
        leaves_grad = np.sum(leaves_grad_matrix * leaves_prob[:,:,i],axis=0)
        leaves_grad_class.append(leaves_grad)

    return np.array(leaves_grad_class)

def _leaves_grad_cos(input_sample, leaves_interval, leaves_prob, lower_bound, upper_bound, grad_fnc=_indicatior_grad_cos):
    data_range = 1.1*upper_bound - 0.9*lower_bound
    indicator_grad ,indicator= grad_fnc(input_sample, leaves_interval, data_range=data_range)
    leaves_grad_matrix = np.ones(indicator.shape)
    leaves_grad_class = []
    for i in range(indicator.shape[1]):
        leaves_grad_matrix[:,i] = indicator_grad[:,i] * np.prod(indicator[:,0:i],axis=1) * np.prod(indicator[:,i+1:],axis=1)
    for i in range(leaves_prob.shape[2]):
        leaves_grad = np.sum(leaves_grad_matrix * leaves_prob[:,:,i],axis=0)
        leaves_grad_class.append(leaves_grad)

    return np.array(leaves_grad_class)

def _leaves_grad_dis(input_sample, leaves_interval, leaves_prob, lower_bound, upper_bound, grad_fnc=_indicatior_grad_dis):
    data_range = 1.1 * upper_bound - 0.9 * lower_bound
    indicator_grad, indicator = grad_fnc(input_sample, leaves_interval, data_range=data_range)
    indicator_grad ,indicator= grad_fnc(input_sample, leaves_interval)
    leaves_grad_matrix = np.ones(indicator.shape)
    leaves_grad_class = []
    for i in range(indicator.shape[1]):
        leaves_grad_matrix[:,i] = indicator_grad[:,i] * np.prod(indicator[:,0:i],axis=1) * np.prod(indicator[:,i+1:],axis=1)
    for i in range(leaves_prob.shape[2]):
        leaves_grad = np.sum(leaves_grad_matrix * leaves_prob[:,:,i],axis=0)
        leaves_grad_class.append(leaves_grad)

    return np.array(leaves_grad_class)




#%%
if __name__ =='__main__':
    
    from read_data import file_to_data
    import joblib
    from RF_to_interval import _parse_forest_interval
    _, test_data, test_label = file_to_data('/Users/zh/py_program/learn_focus/data/cf_wine_data_test.tsv')
    data_range = np.max(test_data,axis=0) - np.min(test_data,axis=0)
    
    model_name = 'wine_forest_depth4_100_lzh_model.pkl'
    model = joblib.load(model_name)
    leaves_interval, leaves_prob = _parse_forest_interval(model)
    D,E,F ,_,_= _indicator_function_leaf_interval(test_data[0], leaves_interval)
    G = D+E+F
    H,_ = _indicatior_grad(test_data[0], leaves_interval)
    H2,HH2 = _indicatior_grad_cos(test_data[0], leaves_interval, data_range)
    H3,_ = _indicatior_grad_dis(test_data[0], leaves_interval, data_range)
    I = _leaves_grad(test_data[0], leaves_interval, leaves_prob)
    I2 = _leaves_grad(test_data[0], leaves_interval, leaves_prob,grad_fnc=_indicatior_grad_cos)
    I3 = _leaves_grad(test_data[0], leaves_interval, leaves_prob,grad_fnc=_indicatior_grad_dis)
    #%%
    
    J = []
    J.append(I[0])
    J.append(I2[0])
    J.append(I3[0])
    J = np.array(J)
    J.reshape((3,-1))
    
