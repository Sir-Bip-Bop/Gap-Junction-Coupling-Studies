import numpy as np 

def phases(data,dt):
    points1 = np.zeros(len(data[:,0]))
    points2 = np.zeros(len(data[:,1]))
    num_points1 = 0
    num_points2 = 0
    for i in range(1,len(data)-2):
        if  (data[i-1,0] < data[i,0] and data[i,0] > data[i+1,0] ) or (data[i-1,0] > data[i,0] and data[i,0] < data[i+1,0]):
            points1[num_points1] = i * dt
            num_points1 = num_points1 +1
        if  (data[i-1,1] < data[i,1] and data[i,1] > data[i+1,1] ) or (data[i-1,1] > data[i,1] and data[i,1] < data[i+1,1]):
            points2[num_points2] = i * dt
            num_points2 = num_points2 +1
        if points1[num_points1] > points2[num_points2]:
            points2[num_points2] = 0
            num_points2 = num_points2 -1
    time_dif = 0
    if len(points1) != len(points2):
        print('error')
    for i in range(0,num_points1-1):
        time_dif = time_dif + (points1[i] - points2[i])
    time_dif = time_dif/num_points1
    return abs(time_dif),float(points1[i])