import argparse
from os.path import join
import pickle
import torch
import numpy as np
import time
from PIL import Image
from skimage.io import imread, imsave
import cv2


import sys
sys.path.append('functions')
from SMPL_Pytorch import rodrigues, get_poseweights, par_to_mesh
from render import mesh2Image, transpose_mesh, scale_mesh
from scale_mask import scale_fixed_height
import neural_renderer as nr


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
        default=8,
        type=int,
        help="Number of views as input [8]",
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
        default=0,
        type=int,
        help="which gpu to train [1]"
    )

    return parser.parse_args()

def generate_rots_poses_betas():
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
    return rots, poses, betas
    

def scale_vert_nr_batch(v):      # v.shape=[batch, 6890, 3]
    v = v - v.min(v.dim()-2)[0].unsqueeze(v.dim()-2)    #[None, :] == torch.unsqueeze( ,dimension 0)
    v /= torch.abs(v).max(1)[0].max(1)[0].reshape(-1,1,1)
    v *= 2
    v -= v.max(v.dim()-2)[0].unsqueeze(v.dim()-2)/2

    # v -= v.min(0)[0][None, :]
    # v /= torch.abs(v).max()
    # v *= 2
    # v -= v.max(0)[0][None,:]/2

    return v

def scale_vert_nr_batch_forLoop(v): # v.shape=[batch, 6890, 3]
    for i in range(v.shape[0]):
        v[i,:,:] = scale_vert_nr(v[i,:,:])
    return v
        


def scale_vert_nr(v):   # v.shape=[6890, 3]
    v = v - v.min(0)[0][None, :]
    # if torch.abs(v).max()>1:
    v = v / torch.abs(v).max()
    v = v * 2
    v = v - v.max(0)[0][None,:]/2
    return v

def main():
    args = parse_args()
    #args.dataset_size = 10
    gender = 'male'
    print('-----------------------------------------------------------')
    print('Gender: ', gender)
    m = pickle.load(open('models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % gender[0]))
    dataset_size = args.dataset_size
    print('Dataset range:', args.start_num, ' - ', dataset_size+args.start_num-1)
    parent_dic = '/home/yifu/Data/test'
    print('Data Path: ',parent_dic)
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print('-----------------------------------------------------------')
    if raw_input('Confirm the above setting? (y/n): ')!='y':
        print('Terminated')
        exit()
    print('Data generation starts')
    print('------------------------')

    for i in range(0,  dataset_size):
        i = i + args.start_num
        rots, poses, betas = generate_rots_poses_betas()
        vertices = par_to_mesh(gender, rots, poses, betas)
   
        # rendering
        vertices_num = 6890
        path = parent_dic
        batch = 1
        img_height = 264
        img_width = 192 
        
        vertices = vertices.squeeze().to("cpu").numpy() # frontal view vertices

        # multi views
        num_view = args.num_views
        camera = torch.zeros(num_view).cuda()

        angle = 2*np.pi / num_view    
        axis = np.array([0, 1, 0])
        

        for view in range (0, num_view):
            name = '%d_%d'%(i, view)

            noise = angle * (np.random.rand()-0.5)
            # noise = 0
            mesh_rot = angle * view + noise 
            v = transpose_mesh(vertices, mesh_rot, axis)    # temp vertices
            camera[view] = mesh_rot
            
            
            # rendering: opendr
            '''
            v_0 = scale_mesh(v, 400, 400, vertices_num=vertices_num)
            mesh2Image(v_0, m['f'], batch, path, name, 400, 400, vertices_num=vertices_num)
            image = Image.open(join(path,name+'.png'))
            output = scale_fixed_height(image,img_height,img_width)
            output_name = join(path, name+'.png')
            output.save(output_name)
            '''

            # rendering: neural renderer
            n_renderer = nr.Renderer(image_size=300, perspective=False, camera_mode='look_at')
            f_nr = torch.from_numpy(m['f'].astype(int)).cuda()
            f_nr = f_nr[None, :, :]

            v_nr = v
            # normalize mesh into a unit cube centered zero (length 2, -1 to 1), see nr/load_obj.py
            v_nr = torch.from_numpy(v_nr).cuda()
            v_nr = scale_vert_nr(v_nr)

            v_nr = torch.unsqueeze(v_nr.float(), 0)
            images = n_renderer(v_nr, f_nr, mode='silhouettes') # silhouettes
            image = images.detach().cpu().numpy()[0]*255
            
            cv2.imwrite(join(path, name+'.png'), image)
            # imsave(join(path, name+'.png'), image)

            # test: rotate mesh using parameter not transpose mesh, compare the generated vert and image
            '''
            rots[0][1] = mesh_rot
            v_test = par_to_mesh(gender, rots, poses, betas)
            v_nr = v_test.squeeze().to("cpu").numpy()
            # print(np.array_equal(v,v_nr))
            # normalize mesh into a unit cube centered zero (length 2, -1 to 1), see nr/load_obj.py
            v_nr = torch.from_numpy(v_nr).cuda()
            v_nr -= v_nr.min(0)[0][None, :]
            v_nr /= torch.abs(v_nr).max()
            v_nr *= 2
            v_nr -= v_nr.max(0)[0][None, :] / 2

            v_nr = torch.unsqueeze(v_nr.float(), 0)
            images = n_renderer(v_nr, f_nr, mode='silhouettes') # silhouettes
            image_cmp = images.detach().cpu().numpy()[0]*255
            print(np.array_equal(image,image_cmp))
            cv2.imwrite(join(path, name+'_par.png'), image)
            '''
            
            
            

        # pickle 
        poses = torch.reshape(poses, (1,-1))

        parameters = torch.cat((rots[0], poses[0], betas[0], camera),0).to("cpu")   # 82 SMPL + num_view
        
        pickle.dump(parameters, open('%s/%d' % (path, i), 'wb'))

        '''
        # mesh generator
        outmesh_path = join(path, '%d.obj'%i)

        with open(outmesh_path, 'w') as fp:
            for v in vertices:
                #fp.write( 'v %f %f %f\n' % ( float(v[0]),float(v[1]), float(v[2])) )
                fp.write( 'v %f %f %f\n' % (v[0], v[1],v[2]))
            for f in m['f']+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
        '''




if __name__ == "__main__":
    main()