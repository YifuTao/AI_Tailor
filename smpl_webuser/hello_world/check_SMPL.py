import render
from render import rendering
import sys
sys.path.append("/home/yifu/workspace/smpl")
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
import pickle
from mesh_generator import generate_mesh
import argparse
import os
from numpy import random
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="generate dataset for training&validation")
    parser.add_argument(
        "--dataset_size",
        default=1000,
        type=int,
        help="size of the whole (train+val) dataset [1000]",
    )
    parser.add_argument(
        "--num_views",
        default=1,
        type=int,
        help="Number of views as input [1]",
    )
    parser.add_argument(
        "--shape_value",
        default=1,
        type=float,
        help="Shape parameter value [1]",
    )
    parser.add_argument(
        "--width",
        default=600,
        type=int,
        help="Output image width[600]",
    )
    parser.add_argument(
        "--height",
        default=600,
        type=int,
        help="Output image height[800]",
    )
    return parser.parse_args()

## generate dataset
args = parse_args()

batch_size = 1
parent = "/home/yifu/workspace/data_smpl"
path = os.path.join(parent, 'test')

## Get some data with three views
dataset = 1
for i in range(0,dataset):
    #betas = np.random.rand(10) * 4 * 2 - 4
    betas = [1.0615,  0.2208,  0.7562,  0.0171,  1.1584,  0.0288,  0.8465,  0.4207,
        -0.5964,  0.4239]

    pose = np.zeros(72)
    #pose[1]=np.pi
    pose[2]=np.pi

    # pose[joint*3 + plane]  plane: 0:xz  1:yz , 2:xy,  
    # x,y axis of frontal view
    # z axis: out of plane of fontal view
    
    pose[16*3 + 2]=-np.pi/4 #shoulder joint
    pose[16*3 + 1]=-np.pi/32
    '''
    pose[17*3 + 2]=np.pi/4  #shoulder joint
    pose[17*3 + 1]=np.pi/32

    pose[1*3 +2] = np.pi/40
    pose[2*3 +2] = -np.pi/40

    pose[50]-=0.2
    pose[53]+=0.2
    '''

    img_name = '%d_0'% (i)
    #img_name = '1'
    parameters = np.append(pose[:],betas[:])
    angle = np.pi
    axis = np.array([0, 1, 0])

    m = load_model('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    # import time
    # since = time.time()
   # for _ in tqdm(range(100)):
    rendering(m, parameters, batch_size, path, img_name, args.width, args.height, angle, axis)
    # print'rendering one image:',time.time()-since
    img_name = '%d_1'% (i)
    #img_name = '0'
    angle = np.pi/2
    axis = np.array([0, 1, 0])
    rendering(m, parameters, batch_size, path, img_name, args.width, args.height, angle, axis)

    img_name = 'id_4'
    generate_mesh(parameters,batch_size,path,img_name)

    # save side view image
    img_name = '%d_1'% i
    #pose[0]=0

    
    '''
    pose[0]=np.pi
    parameters = np.append(pose[:],betas[:])
    rendering(parameters,batch_size,path,img_name,args.width,args.height)

    
    # save thrid view image
    
    img_name = '%d_2'% i
    pose[1]=np.pi/4
    parameters = np.append(pose[:],betas[:])
    rendering(parameters,batch_size,path,img_name,args.width,args.height)
    '''


## check shape 
'''
for i in range(0,10):


    # generate parameters


    pose = np.zeros(72)
    betas = np.zeros(10)

    value = args.shape_value
    betas[i] = value
 
    img_name = 'shape_%d=%d'% (i,value)
    parameters = np.append(pose[:],betas[:])
    rendering(parameters,batch_size,path,img_name)
    generate_mesh(parameters,batch_size,path,img_name)
'''