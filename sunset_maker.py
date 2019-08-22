import numpy as np
import skimage
#Import Sklearn modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import load

colors = ['red', 'green', 'blue']

red = load('red.joblib')
green = load('green.joblib')
blue = load('blue.joblib')

width = 200
height = 180

array_size = width * height

seed_image = skimage.io.imread('sunsets//'+'3qteasg.jpeg')
output = []
for i in range(15):
    output.append(seed_image[0][i]/255.0)
    
del seed_image


for i in range(15, array_size):
    red_x = []
    green_x = []
    blue_x = []
    for j in range(15):
        red_x.append(output[i-15+j][0])
        green_x.append(output[i-15+j][1])
        blue_x.append(output[i-15+j][2])
    r= red.predict([red_x])
    g= red.predict([green_x])
    b= red.predict([blue_x])
    output.append([r[0],g[0],b[0]])