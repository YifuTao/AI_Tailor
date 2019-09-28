import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import numpy as np
import pickle
import sys
print(sys.version)
# Now read the smpl model.
with open('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    data = pickle.load(f)
    #print(data['v_template'][16])
    #print(data['v_template'][17])
    Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
    X,Y,Z = [Vertices[:,0], Vertices[:,1],Vertices[:,2]]

def smpl_view_set_axis_full_body(ax,azimuth=0):
    ## Manually set axis 
    ax.view_init(0, azimuth)
    max_range = 0.55
    ax.set_xlim( - max_range,   max_range)
    ax.set_ylim( - max_range,   max_range)
    ax.set_zlim( -0.2 - max_range,   -0.2 + max_range)
    ax.axis('off')
    
def smpl_view_set_axis_face(ax, azimuth=0):
    ## Manually set axis 
    ax.view_init(0, azimuth)
    max_range = 0.1
    ax.set_xlim( - max_range,   max_range)
    ax.set_ylim( - max_range,   max_range)
    ax.set_zlim( 0.45 - max_range,   0.45 + max_range)
    ax.axis('off')

## Now let's rotate around the model and zoom into the face.

#fig = plt.figure(figsize=[16,4])
fig = plt.figure(figsize=[8,16])

#ax = fig.add_subplot(141, projection='3d')
ax = fig.gca(projection='3d')
#ax.scatter(Z[2],X[2],Y[2],s=0.02,marker='o',c='r')
#ax.scatter(Z[:10],X[0:10],Y[0:10],s=0.02,marker='*',c='r')
ax.scatter(Z,X,Y,s=0.02,c='k')
#smpl_view_set_axis_full_body(ax)

 
A=(3512,135,0,3763,251,3646,3673,133,3771,159,3680,335,259,168,158,3764,254,3670,3645,169,158,3764,254,3670,3645,169,3679,3784,271,134,385)
B=(3920,151,218,425,3796,3731,3734,284,3664,222,219,3797,152)
D=(4740,6420,6485,757,6325,1287,1255,4241,1434,1283,2868,4907,6326,6481,753,4766,3039,2864,4825,6482,2866,5227,3506,1835,6477,1286,1349,6364,6329,4245,2960,4765,2959,4770,1418,6417,5295,611,1760,4892,1834,758,4101,5226,4248,1254,740,943,3042,739,3040,4737,3041,3044,760)
E=(3500,4332,3021,4402,917,1782,5245,1779,705,5244,4193,830,4403,916,2929,6388,4317,2928,831,1336,4812,846,4166,678,6377,6376,6374,4425,2917,939,2918,6370,4167,679,2916,2911)
F=(6540,3118,6539,3084,4350,4919,6541,864,915,6557,6509,4351,863,4692,932,1205,4418,4690,4983,3136,1447,6558,3117,3119,1807,6559,1513,3116,4399,4984,3138,3137,4920,1446,4927,1454)
G=(2102,2103,2104,2148,2105,2108,2110,2106,2109,2206,2208,2112,2241,2230,2235)
H=(1310,1390,1314,1376,1384,1377,1391,1392,1387,1234,1737,1386,1231,1387,1398,1395,629,626,788)
I=(1560,1563,1599,1721,1685,1688,1602,1726,1581,1555,1552,1549,1578,1692,1546,1550)
L=(4450,4335,1359,4835,4389,4362,4757,4971,4707,4363,4337,4708,4422,4385,4336,4394,4421,4386,4395,4358,4839,4448,4926,1453)
M=(1100,1087,1088,1528,1097,1092,1096,1091,1464,1469,1467,1155,1371,1099,1103)
N=(3210,3208,3201,3204,3207,3205,3202,3200,3203,3198,3199,3326,3325,3209,3324)

height=(6677,411)
#for i in range(0,6890,10):

#    ax.text(Z[i],X[i],Y[i],i,color='red', fontsize=5)
#    ax.scatter(Z[i],X[i],Y[i],s=0.02,marker='*',c='r')

for i in height:
    #ax.annotate(i,(X[i],Y[i],Z[i]))
    ax.text(Z[i],X[i],Y[i],i,color='green', fontsize=6)
    ax.scatter(Z[i],X[i],Y[i],s=0.06,marker='*',c='g')
    print(i,X[i],Y[i],Z[i])

smpl_view_set_axis_full_body(ax)

#ax = fig.add_subplot(142, projection='3d')
#ax.scatter(Z,X,Y,s=0.02,c='k')
#smpl_view_set_axis_full_body(ax,45)

#ax = fig.add_subplot(143, projection='3d')
#ax.scatter(Z,X,Y,s=0.02,c='k')
#smpl_view_set_axis_full_body(ax,90)

#ax = fig.add_subplot(144, projection='3d')
#ax.scatter(Z,X,Y,s=0.2,c='k')
#smpl_view_set_axis_face(ax,-40)

plt.show()
