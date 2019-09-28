import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import torch
import numpy as np
import pickle
import os
import sys
sys.path.append("/home/yifu/workspace/smpl")


def dist(a, b):
    sq_sum = (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2
    return torch.sqrt(sq_sum)
    #return np.sqrt((float(a[0])-float(b[0]))**2+(float(a[1])-float(b[1]))**2+(float(a[2])-float(b[2]))**2)

   


def calculate_geo_distance(list_name, X, Y, Z):
    all_dist = 0
    for i, element in enumerate(list_name):
        point_A = (X[:,list_name[i]], Y[:,list_name[i]], Z[:,list_name[i]])
        point_B = (X[:,list_name[i-1]], Y[:,list_name[i-1]], Z[:,list_name[i-1]])
        dist_AB = dist(point_A, point_B)
        all_dist = all_dist+dist_AB
    return all_dist




def get_measurements(m, parameters):  # 82 parameters
     m.pose[:] = parameters[:72]
     m.betas[:] = parameters[72:]
     Vertices = m.v_shaped  # (6890, 3)
     X, Y, Z = [Vertices[:, 0], Vertices[:, 1], Vertices[:, 2]]

     h, w, c, n, a = vertex2measurements(X, Y, Z)
     return h, w, c, n, a

def vertex2measurements(X,Y,Z):
     A=(3646,3771,335,259,133,134,159,158,169,168,271,254,385,3764,3763,3784,3680,3679,3670,3673,3645)

     B = (1331,215,216,440,441,452,218,219,222,425,453,829,3944,3921,3920,3734,3731,3730,3943,3935,3934,
     3728,3729,4807,3068) 

     D = (3015,4238,4237,4718,5295,4908,4910,6418,4246,4248,4229,4228,4892,5226,6327,6328,4098,4100,6490,
     6491,6487,6488,6489,4428,4737,6332,3076,2870,1254,943,3042,3040,3041,3044,3043,610,2867,2865,1419,
     740,739,760,2958,1435,1437,1235,749,752)

     E = (3503,4347,4346,4689,4343,4342,4922,4921,4425,4332,4317,4316,4331,4330,4373,6389,6388,5244,5245,
     3021,1782,1779,2929,2928,886,845,844,831,830,846,939,1449,1448,857,856,1203,860,859)

     F=(4350,4351,4692,4399,4984,4927,4920,4919,6509,6557,6558,6559,6540,6539,6541,3119,3118,3117,3116,
     3138,3137,3136,3084,1447,1446,1454,1513,915,1205,863,864,1807)

     G=(2102,2103,2104,2105,2148,2108,2106,2206,2208,2241,2235,2230,2112,2109,2110)

     H=(1310,1314,1376,1377,1391,1392,1390,1387,1234,1231,1386,1384,1737,1398,1395,629,626,788)

     I=(1560,1726,1552,1555,1581,1578,1692,1550,1546,1549,1602,1599,1685,1688,1721,1563)

     L=(4422,4421,4926,4449,4397,4396,4838,4393,4392,4443,4391,4390,4388,4387,4448,4386,4385)

     M=(1100,1097,1096,1469,1467,1464,1091,1092,1528,1088,1087,1155,1371,1103,1099)

     N=(3210,3201,3200,3203,3202,3205,3204,3207,3208,3326,3325,3324,3209,3199,3198)

     #arm = (3010, 2208)
     arm = (6470,4790,5670)

     #height = float(Y[:,411] - Y[:,6677])
     height = (Y[:,411] - Y[:,6677])

     waist = calculate_geo_distance(E, X, Y, Z)

     chest = calculate_geo_distance(D, X, Y, Z)

     neck = calculate_geo_distance(B, X, Y, Z)

     point_arm1 = (X[:,arm[0]], Y[:,arm[0]], Z[:,arm[0]])
     point_arm2 = (X[:,arm[1]], Y[:,arm[1]], Z[:,arm[1]])
     point_arm3 = (X[:,arm[2]], Y[:,arm[2]], Z[:,arm[2]])
     #arm = dist(point_arm1, point_arm2)
     arm = dist(point_arm1, point_arm2)+dist(point_arm2, point_arm3)

     return height, waist, chest, neck, arm
