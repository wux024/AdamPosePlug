#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2022/8/21 16:23
@description:
"""
import numpy as np

# Calculate the derivative of the sequence
def cal_deriv(x, y):  # x, y is list
    diff_x = []  # dx
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)
    diff_y = []  # dy
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)
    slopes = []  # slope
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])
    deriv = []  # dy/dx
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))
    deriv.insert(0, slopes[0])  # dy/dx-
    deriv.append(slopes[-1])  # dy/dx+
    # for i in deriv:  # test
    #     print(i)
    return np.array(deriv)  # return dy/dx