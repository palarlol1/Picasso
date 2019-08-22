#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:56:56 2019

@author: richard
"""
import numpy as np
import os
import skimage
#Import Keras modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


np.random.seed(19)

seq_length = 5
sunsets = []

for filename in os.listdir('sunsets'):
    sunsets.append(skimage.io.imread('sunsets//'+filename))

red_x = []
red_y = []
blue_x = []
blue_y = []
green_x = []
green_y = []
for i in range(len(sunsets)):
    for row in range(len(sunsets[i])):
        for col in range(len(sunsets[i][row])):
            rgb_list = sunsets[i][row][col]
            r= rgb_list[0]/255.0
            g = rgb_list[1]/255.0
            b = rgb_list[2]/255.0
            red_x.append([row/len(sunsets[i]),col/len(sunsets[i][row])])
            blue_x.append([row/len(sunsets[i]),col/len(sunsets[i][row])])
            green_x.append([row/len(sunsets[i]),col/len(sunsets[i][row])])
            red_y.append(r)
            green_y.append(g)
            blue_y.append(b)

'''
#Convert to numpy arrays
red = np.array(red)
green = np.array(green)
blue = np.array(blue)

#Going to set up our training locations now


for i in range(len(red) - seq_length):
    red_x.append(red[i:i+seq_length])
    blue_x.append(blue[i:i+seq_length])
    green_x.append(green[i:i+seq_length])
    red_y.append(red[i+seq_length])
    blue_y.append(blue[i+seq_length])
    green_y.append(green[i+seq_length])

del red
del blue
del green
'''

red_x = np.array(red_x)
blue_x = np.array(blue_x)
green_x = np.array(green_x)
red_y = np.array(red_y)
blue_y = np.array(blue_y)
green_y = np.array(green_y)


print("Done Processing data, there are ", red_x.shape)
red_train_x,red_test_x, red_train_y, red_test_y = train_test_split(red_x, red_y, test_size = .25)
green_train_x,green_test_x, green_train_y, green_test_y = train_test_split(green_x, green_y, test_size = .25)
blue_train_x,blue_test_x, blue_train_y, blue_test_y = train_test_split(blue_x, blue_y, test_size = .25)


red_model = RandomForestRegressor(n_estimators = 19, max_depth = 3)

green_model = RandomForestRegressor(n_estimators = 19, max_depth = 2)

blue_model = RandomForestRegressor(n_estimators = 49, max_depth = 3)


red_model.fit(red_train_x, red_train_y)
print("Red Done")
blue_model.fit(blue_train_x, blue_train_y)
print("Gren Done")
green_model.fit(green_train_x, green_train_y)
print("Blue Done")

print(red_model.score(red_test_x, red_test_y))
print(blue_model.score(blue_test_x, blue_test_y))
print(green_model.score(green_test_x, green_test_y))
