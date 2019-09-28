from calculate_circumference import get_measurements
import argparse
from mesh_generator import generate_mesh
import pickle
from smpl_webuser.serialization import load_model
from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
import numpy as np
import render
from render import rendering
from PIL import Image
from scale_mask import scale_fixed_height
from os.path import join
import sys
sys.path.append("/home/yifu/workspace/smpl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="generate dataset for training&validation")
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
        "--start_num",
        default=0,
        type=int,
        help="Starting number (last total dataset_size) [0]",
    )
    parser.add_argument(
        "--width",
        default=400,
        type=int,
        help="Output image width[600]",
    )
    parser.add_argument(
        "--height",
        default=400,
        type=int,
        help="Output image height[800]",
    )
    parser.add_argument(
        "--output_height",
        default=264,
        type=int,
        help="Output image height [264]",
    )
    parser.add_argument(
        "--output_width",
        default=192,
        type=int,
        help="Output image width [192]",
    )
    return parser.parse_args()


def get_height(pose, betas):
    m = load_model('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    #height = (6677,411)
    m.pose[:] = pose
    m.betas[:] = betas[:]
    # rint(m.v_shaped[411][1])
    # print(m.v_shaped[6677][1])
    # print('height:')
    height = m.v_shaped[411][1] - m.v_shaped[6677][1]
    return height
    #print(m.v_shaped[411][1] - m.v_shaped[6677][1])
    # print()


args = parse_args()
gender = 'male'    #   female
m = load_model('../../models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % gender[0])
dataset_size = args.dataset_size
train_size = int(dataset_size*0.95)
val_size = int(dataset_size*0.05)
parent = join("/home/yifu/workspace/data_smpl/A_pose_5", gender,'noisy_asymm')
#parent = join('/home/yifu/workspace/data_smpl/test')

for i in range(0,  dataset_size): 
    if (i < train_size):
        path = join(parent, 'train')
        i = i + int(args.start_num*0.95)
    else:
        path = join(parent, 'val')
        i = i-train_size
        i = i + int(args.start_num*0.05)

    batch_size = 1

    # generate parameters
    pose = np.zeros(72)
    pose[2] = np.pi

    # pose[joint*3 + plane]  plane: 0:xz  1:yz , 2:xy,
    # x,y axis of frontal view
    # z axis: out of plane of fontal view
    pose[16*3 + 2] = -np.pi/4  # shoulder joint
    pose[16*3 + 1] = -np.pi/32
    pose[17*3 + 2] = np.pi/4  # shoulder joint
    pose[17*3 + 1] = np.pi/32
    
    variation = np.random.rand()*0.3*2-0.3
    pose[50] += variation
    variation = np.random.rand()*0.3*2-0.3
    pose[53] -= variation
    
    pose[1*3 + 2] = np.pi/40
    pose[2*3 + 2] = -np.pi/40
    
    variation = np.random.rand()*np.pi/40*2 - np.pi/40
    pose[5] += variation
    variation = np.random.rand()*np.pi/40*2 - np.pi/40
    pose[8] += variation

    for j in range(3, 72):
        variation = 0.02
        pose[j] += np.random.rand()*variation * 2 - variation
    
    variation = 4
    betas = np.random.rand(10) * variation * 2 - variation
    # betas[0] = np.random.rand() * 8 * 2 -8
    variation = 2
    betas[1] = np.random.rand() * variation * 2 - variation
    # betas[3] = np.random.rand() * 8 * 2 -8
    # betas[6] = np.random.rand() * 8 * 2 -8
    # print(betas)

    parameters = np.append(pose[:], betas[:])
    # save frontal view image
    img_name = '%d_0' % i
    parameters = np.append(pose[:], betas[:])
    angle = np.pi
    axis = np.array([0, 1, 0])
    rendering(m, parameters, batch_size, path, img_name,
              args.width, args.height, angle, axis)
    image = Image.open(join(path,img_name+'.png'))
    output = scale_fixed_height(image,args.output_height,args.output_width)
    output_name = join(path, img_name+'.png')
    output.save(output_name)


    # save side view image
    img_name = '%d_1' % i
    angle = np.pi/2
    axis = np.array([0, 1, 0])
    rendering(m, parameters, batch_size, path, img_name,
              args.width, args.height, angle, axis)
    image = Image.open(join(path,img_name+'.png'))
    output = scale_fixed_height(image,args.output_height,args.output_width)
    output_name = join(path, img_name+'.png')
    output.save(output_name)
    # drop off the 3 rot parameters
    # parameters = np.append(pose[3:], betas[:])
    '''
    h, w, c, n, a = get_measurements(m, parameters)

    parameters = np.append(
        parameters, [w/h*3, c/h*3, n/h*3, a/h*3])    # 86 param
    '''
    pickle.dump(parameters, open('%s/%d' % (path, i), 'wb'))
