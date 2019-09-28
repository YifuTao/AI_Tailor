
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import pickle
import os
import sys
sys.path.append("/home/yifu/workspace/smpl")


def dist(x, y):
    return np.sqrt((float(x[0])-float(y[0]))**2+(float(x[1])-float(y[1]))**2+(float(x[2])-float(y[2]))**2)


def calculate_geo_distance(list_name, X, Y, Z):
    all_dist = 0
    for i, element in enumerate(list_name):
        point_A = (X[list_name[i]], Y[list_name[i]], Z[list_name[i]])
        point_B = (X[list_name[i-1]], Y[list_name[i-1]], Z[list_name[i-1]])
        dist_AB = dist(point_A, point_B)
        all_dist = all_dist+dist_AB
    return all_dist


def smpl_view_set_axis_full_body(ax, azimuth=0):
    # Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.55
    ax.set_xlim(- max_range,   max_range)
    ax.set_ylim(- max_range,   max_range)
    ax.set_zlim(-0.2 - max_range,   -0.2 + max_range)
    ax.axis('off')


def smpl_view_set_axis_face(ax, azimuth=0):
    # Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.1
    ax.set_xlim(- max_range,   max_range)
    ax.set_ylim(- max_range,   max_range)
    ax.set_zlim(0.45 - max_range,   0.45 + max_range)
    ax.axis('off')


def get_measurements(m, parameters):  # 82 parameters
     m.pose[:] = parameters[:72]
     m.betas[:] = parameters[72:]
     Vertices = m.v_shaped  # (6890, 3)
     X, Y, Z = [Vertices[:, 0], Vertices[:, 1], Vertices[:, 2]]

     h, w, c, n, a = vertex2measurements(X, Y, Z)
     return h, w, c, n, a

def vertex2measurements(X,Y,Z):
     # new

     A = (3512, 135, 0, 3763, 251, 3646, 3673, 133, 3771, 159, 3680, 335, 259, 168, 158,
          3764, 254, 3670, 3645, 169, 158, 3764, 254, 3670, 3645, 169, 3679, 3784, 271, 134, 385)
     '''
     # old vertexes
     B = (3664, 3665, 3797, 3796, 3839, 334, 284, 151, 152, 218,
          219, 222, 425, 426, 453, 3944, 3921, 3920, 3734, 3731)
     D = (1835, 1434, 2960, 2959, 757, 758, 739, 740, 1418, 1760, 2866, 2864, 2868, 611, 3039, 3044, 3041, 3040, 3042, 943, 1349, 1255, 1254, 3506, 4737, 4740, 4825,
          6477, 6482, 6481, 6485, 4101, 6329, 6325, 6326, 5227, 4766, 4765, 4770, 4241, 4245, 6417, 6420, 4907, 4908, 5295, 6391, 6390, 6476, 3026, 2931, 2932, 2933)
     E = (679, 678, 705, 939, 846, 830, 831, 2917, 2918, 2911, 2928, 2929, 1779, 1782, 3021, 5245, 5244, 6388,
          6370, 6376, 6377, 6374, 4317, 4332, 4425, 4193, 4166, 4167, 4403, 4402, 4812, 3500, 1336, 917, 916)
     '''
     # new vertexes
     B = (1331,215,216,440,441,452,218,219,222,425,453,829,3944,3921,3920,3734,3731,3730,
          3943,3935,3934,3728,3729,4807,3068) 
     D = (3015,4238,4237,4718,5295,4908,4910,6418,4246,4248,4229,4228,4892,5226,6327,6328,
          4098,4100,6490,6491,6487,6488,6489,4428,4737,6332,3076,2870,1254,943,3042,3040,
          3041,3044,3043,610,2867,2865,1419,740,739,760,2958,1435,1437,1235,749,752)
     E = (3503,4347,4346,4689,4343,4342,4922,4921,4425,4332,4317,4316,4331,4330,4373,6389,
          6388,5244,5245,3021,1782,1779,2929,2928,886,845,844,831,830,846,939,1449,1448,
          857,856,1203,860,859)

     F = (6540, 3118, 6539, 3084, 4350, 4919, 6541, 864, 915, 6557, 6509, 4351, 863, 4692, 932, 1205, 4418, 4690,
          4983, 3136, 1447, 6558, 3117, 3119, 1807, 6559, 1513, 3116, 4399, 4984, 3138, 3137, 4920, 1446, 4927, 1454)
     G = (2102, 2103, 2104, 2148, 2105, 2108, 2110,
          2106, 2109, 2206, 2208, 2112, 2241, 2230, 2235)
     H = (1310, 1390, 1314, 1376, 1384, 1377, 1391, 1392, 1387,
          1234, 1737, 1386, 1231, 1387, 1398, 1395, 629, 626, 788)
     I = (1560, 1563, 1599, 1721, 1685, 1688, 1602, 1726,
          1581, 1555, 1552, 1549, 1578, 1692, 1546, 1550)
     L = (4450, 4335, 1359, 4835, 4389, 4362, 4757, 4971, 4707, 4363, 4337, 4708,
          4422, 4385, 4336, 4394, 4421, 4386, 4395, 4358, 4839, 4448, 4926, 1453)
     M = (1100, 1087, 1088, 1528, 1097, 1092, 1096,
          1091, 1464, 1469, 1467, 1155, 1371, 1099, 1103)
     N = (3210, 3208, 3201, 3204, 3207, 3205, 3202,
          3200, 3203, 3198, 3199, 3326, 3325, 3209, 3324)

     #arm = (3010, 2208)
     arm = (6470,4790,5670)

     height = float(Y[411] - Y[6677])

     waist = calculate_geo_distance(E, X, Y, Z)

     chest = calculate_geo_distance(D, X, Y, Z)

     neck = calculate_geo_distance(B, X, Y, Z)

     point_arm1 = (X[arm[0]], Y[arm[0]], Z[arm[0]])
     point_arm2 = (X[arm[1]], Y[arm[1]], Z[arm[1]])
     point_arm3 = (X[arm[2]], Y[arm[2]], Z[arm[2]])
     #arm = dist(point_arm1, point_arm2)
     arm = dist(point_arm1, point_arm2)+dist(point_arm2, point_arm3)

     return height, waist, chest, neck, arm

'''
m = load_model('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')


parent_dic = "/home/yifu/workspace/data_smpl"
path = os.path.join(parent_dic, 'real_human', 'check_scaled')

dataset_size = 9
real_height = [1.68, 1.68, 1.70, 1.72, 1.6, 1.81, 1.72, 1.79, 1.83]
if len(real_height) != dataset_size:
    print('wrong real_height!')

for n in range(dataset_size):
    print(n)
    #pickle_path = '%s/%d_%s'% (path, n, )
    pickle_path = os.path.join(path, '%d' % n)
    infile = open(pickle_path, 'rb')
    parameters = pickle.load(infile)
    parameters = parameters[:79]
    parameters = np.append(np.zeros(3), parameters)  # 82

    m.pose[:] = parameters[:72]
    m.betas[:] = parameters[72:]
    Vertices = m.v_shaped  # (6890, 3)
    X, Y, Z = [Vertices[:, 0], Vertices[:, 1], Vertices[:, 2]]

    A = (3512, 135, 0, 3763, 251, 3646, 3673, 133, 3771, 159, 3680, 335, 259, 168, 158,
         3764, 254, 3670, 3645, 169, 158, 3764, 254, 3670, 3645, 169, 3679, 3784, 271, 134, 385)
    B = (3664, 3665, 3797, 3796, 3839, 334, 284, 151, 152, 218,
         219, 222, 425, 426, 453, 3944, 3921, 3920, 3734, 3731)
    D = (1835, 1434, 2960, 2959, 757, 758, 739, 740, 1418, 1760, 2866, 2864, 2868, 611, 3039, 3044, 3041, 3040, 3042, 943, 1349, 1255, 1254, 3506, 4737, 4740, 4825,
         6477, 6482, 6481, 6485, 4101, 6329, 6325, 6326, 5227, 4766, 4765, 4770, 4241, 4245, 6417, 6420, 4907, 4908, 5295, 6391, 6390, 6476, 3026, 2931, 2932, 2933)
    E = (679, 678, 705, 939, 846, 830, 831, 2917, 2918, 2911, 2928, 2929, 1779, 1782, 3021, 5245, 5244, 6388,
         6370, 6376, 6377, 6374, 4317, 4332, 4425, 4193, 4166, 4167, 4403, 4402, 4812, 3500, 1336, 917, 916)
    F = (6540, 3118, 6539, 3084, 4350, 4919, 6541, 864, 915, 6557, 6509, 4351, 863, 4692, 932, 1205, 4418, 4690,
         4983, 3136, 1447, 6558, 3117, 3119, 1807, 6559, 1513, 3116, 4399, 4984, 3138, 3137, 4920, 1446, 4927, 1454)
    G = (2102, 2103, 2104, 2148, 2105, 2108, 2110,
         2106, 2109, 2206, 2208, 2112, 2241, 2230, 2235)
    H = (1310, 1390, 1314, 1376, 1384, 1377, 1391, 1392, 1387,
         1234, 1737, 1386, 1231, 1387, 1398, 1395, 629, 626, 788)
    I = (1560, 1563, 1599, 1721, 1685, 1688, 1602, 1726,
         1581, 1555, 1552, 1549, 1578, 1692, 1546, 1550)
    L = (4450, 4335, 1359, 4835, 4389, 4362, 4757, 4971, 4707, 4363, 4337, 4708,
         4422, 4385, 4336, 4394, 4421, 4386, 4395, 4358, 4839, 4448, 4926, 1453)
    M = (1100, 1087, 1088, 1528, 1097, 1092, 1096,
         1091, 1464, 1469, 1467, 1155, 1371, 1099, 1103)
    N = (3210, 3208, 3201, 3204, 3207, 3205, 3202,
         3200, 3203, 3198, 3199, 3326, 3325, 3209, 3324)

    height = (6677, 411)
    arm = (3010, 2208)

    print('Height:')
    # print(Y[height[1]]-Y[height[0]])
    print(real_height[n])
    img_height = Y[height[1]]-Y[height[0]]
    r_height = real_height[n]
    ratio = r_height / img_height
    ratio = float(ratio)

    all_dist = calculate_geo_distance(E)
    print('Waist circumference:')
    print(all_dist * ratio)

    all_dist = calculate_geo_distance(D)
    print('Chest circumference:')
    print(all_dist * ratio)

    all_dist = calculate_geo_distance(B)
    print('Neck circumference:')
    print(all_dist * ratio)

    point_arm1 = (X[arm[0]], Y[arm[0]], Z[arm[0]])
    point_arm2 = (X[arm[1]], Y[arm[1]], Z[arm[1]])
    arm_dist = dist(point_arm1, point_arm2)
    print('Arm length:')
    print(arm_dist * ratio)

    print
'''
