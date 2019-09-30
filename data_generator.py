import argparse
from os.path import join
import pickle
import torch
import numpy as np
import time
from PIL import Image

import sys
sys.path.append('functions')
from SMPL_Pytorch import rodrigues, get_poseweights, par_to_mesh
from render import mesh2Image, transpose_mesh, scale_mesh
from scale_mask import scale_fixed_height



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
    parser.add_argument(
        "--gpu",
        default=1,
        type=int,
        help="which gpu to train [1]"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    #args.dataset_size = 10
    gender = 'male'
    print('-----------------------------------------------------------')
    print'Gender: ', gender
    m = pickle.load(open('models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % gender[0]))
    dataset_size = args.dataset_size
    print'Dataset range:', args.start_num, ' - ', dataset_size+args.start_num-1
    parent_dic = '/home/yifu/workspace/data/synthetic/noisy/train'
    print 'Data Path: ',parent_dic
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print('-----------------------------------------------------------')
    if raw_input('Confirm the above setting? (yes/no): ')!='yes':
        print('Terminated')
        exit()
    print'Data generation starts'
    print('------------------------')

    for i in range(0,  dataset_size):
        i = i + args.start_num

        rots = (torch.zeros(1,3)).cuda()    # global rotation
        poses = (torch.zeros(1,23,3)).cuda()   # joints
        variation = 3
        betas = (torch.rand(1,10) * variation*2 - variation).cuda()    # shapes

        poses[0][17 - 1][2] = np.pi/4   # right shoulder joint
        poses[0][16 - 1][2] = - np.pi/4   # left shoulder joint
        
        variation = 0.3
        poses[0][17 - 1][2] += (torch.rand(1)[0] * variation*2 - variation).cuda()
        
        poses[0][16 - 1][2] = - poses[0][17 - 1][2]
        
        variation = 0.15
        poses[0][16 - 1][2] += (torch.rand(1)[0] * variation*2 - variation).cuda()

        poses[0][1 - 1][2] = np.pi/40
        variation = np.pi/40
        poses[0][1 - 1][2] += (torch.rand(1)[0] * variation*2 - variation).cuda()
        poses[0][2 - 1][2] = -np.pi/40
        variation = np.pi/40
        poses[0][2 - 1][2] += (torch.rand(1)[0] * variation*2 - variation).cuda()

        variation = 0.03
        for j in range(0,23):
            for k in range(0,3):
                poses[0][j][k] += torch.rand(1)[0] *variation*2 - variation
        

        vertices = par_to_mesh(gender, rots, poses, betas)
        
        # rendering
        vertices_num = 6890
        path = parent_dic
        batch = 1
        img_height = 264
        img_width = 192 

        # frontal view
        name = '%d_0'%i
        v = vertices.squeeze().to("cpu").numpy()
        angle = np.pi
        axis = np.array([0, 0, 1])
        v = transpose_mesh(v, angle, axis)
        angle = np.pi
        axis = np.array([0, 1, 0])
        v = transpose_mesh(v, angle, axis)
        v_0 = scale_mesh(v, 400, 400, vertices_num=vertices_num)
        mesh2Image(v_0, m['f'], batch, path, name, 400, 400, vertices_num=vertices_num)
        image = Image.open(join(path,name+'.png'))
        output = scale_fixed_height(image,img_height,img_width)
        output_name = join(path, name+'.png')
        output.save(output_name)
        # side view
        name = '%d_1'%i
        angle = np.pi/2
        axis = np.array([0, -1, 0])
        v = transpose_mesh(v, angle, axis)
        v_1 = scale_mesh(v, 400, 400, vertices_num=vertices_num)
        mesh2Image(v_1, m['f'], batch, path, name, 400, 400, vertices_num=vertices_num)
        image = Image.open(join(path,name+'.png'))
        output = scale_fixed_height(image,img_height,img_width)
        output_name = join(path, name+'.png')
        output.save(output_name)
        # pickle 
        poses = torch.reshape(poses, (1,-1))
        parameters = torch.cat((rots[0], poses[0], betas[0]),0).to("cpu")
        
        pickle.dump(parameters, open('%s/%d' % (path, i), 'wb'))
        '''
        # mesh generator
        outmesh_path = join(path, '%d.obj'%i)

        with open(outmesh_path, 'w') as fp:
            for v in vertices[0]:
                #fp.write( 'v %f %f %f\n' % ( float(v[0]),float(v[1]), float(v[2])) )
                fp.write( 'v %f %f %f\n' % (v[0], v[1],v[2]))
            for f in m['f']+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
        '''




if __name__ == "__main__":
    main()