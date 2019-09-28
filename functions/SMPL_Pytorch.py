import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import pickle
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


bases_num = 10  # number of base shapes
mesh_num = 6890 # mesh number
keypoints_num = 24 # joint


def rodrigues(r):
    theta = torch.sqrt(torch.sum(torch.pow(r, 2),1))

    def S(n_):
        ns = torch.split(n_, 1, 1)
        Sn_ = torch.cat([torch.zeros_like(ns[0]),-ns[2],ns[1],ns[2],torch.zeros_like(ns[0]),-ns[0],-ns[1],ns[0],torch.zeros_like(ns[0])], 1)
        # maybe cat instead ot stack
        Sn_ = Sn_.view(-1, 3, 3)
        return Sn_

    n = r/(theta.view(-1, 1))
    Sn = S(n)

    #R = torch.eye(3).unsqueeze(0) + torch.sin(theta).view(-1, 1, 1)*Sn\
    #        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)

    I3 = Variable(torch.eye(3).unsqueeze(0).cuda())

    R = I3 \
        + torch.sin(theta).view(-1, 1, 1)*Sn \
        + (1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn, Sn)

    Sr = S(r)
    theta2 = theta**2
    R2 = I3 + (1.-theta2.view(-1,1,1)/6.)*Sr\
        + (.5-theta2.view(-1,1,1)/24.)*torch.matmul(Sr,Sr)

    #idx = np.argwhere(np.asarray(theta<1e-30))
    idx = np.argwhere((theta<1e-30).data.cpu().numpy())

    if (idx.size):
        R[idx,:,:] = R2[idx,:,:]

    return R,Sn

def get_poseweights(poses, bsize):

    # pose: batch x 24 x 3
    pose_matrix, _ = rodrigues(poses[:,1:,:].contiguous().view(-1,3))
    #pose_matrix, _ = rodrigues(poses.view(-1,3))
    pose_matrix = pose_matrix - Variable(torch.from_numpy(np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0),bsize*(keypoints_num-1),axis=0)).cuda())
    pose_matrix = pose_matrix.view(bsize, -1)
    return pose_matrix


## main function 
## SMPL in Pytorch produce mesh that is a shifted version of SMPL
## So the vertex has a constant offset
## X -0.0262    Y -0.0078   Z -0.0846
def par_to_mesh(gender, rots, poses, betas):

    
    dd = pickle.load(open('models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl'%gender[0], 'rb'))

    kintree_table = dd['kintree_table']

    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}
    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}

    mesh_mu = Variable(torch.from_numpy(np.expand_dims(dd['v_template'], 0).astype(np.float32)).cuda()) # zero mean
    mesh_pca = Variable(torch.from_numpy(np.expand_dims(dd['shapedirs'], 0).astype(np.float32)).cuda())
    posedirs = Variable(torch.from_numpy(np.expand_dims(dd['posedirs'], 0).astype(np.float32)).cuda())
    J_regressor = Variable(torch.from_numpy(np.expand_dims(dd['J_regressor'].todense(), 0).astype(np.float32)).cuda())
    weights = Variable(torch.from_numpy(np.expand_dims(dd['weights'], 0).astype(np.float32)).cuda())
    root_rot = Variable(torch.FloatTensor([0,0.,0.]).unsqueeze(0).cuda())

    batch_size = rots.size(0)


    #poses = (hands_mean + torch.matmul(poses.unsqueeze(1), hands_components).squeeze(1)).view(batch_size,keypoints_num-1,3)
    #poses = torch.cat((poses[:,:3].contiguous().view(batch_size,1,3),poses_),1)

    poses = poses.contiguous().view(batch_size, keypoints_num-1, 3)
    poses = torch.cat((root_rot.repeat(batch_size,1).view(batch_size,1,3),poses),1)

    #a = torch.matmul(betas.unsqueeze(1),mesh_pca.repeat(batch_size,1,1,1).permute(0,3,1,2).contiguous().view(batch_size,bases_num,-1))
    '''
    print(betas.unsqueeze(1).shape)
    print(mesh_pca.repeat(batch_size,1,1,1).permute(0,3,1,2).contiguous().view(batch_size,bases_num,-1).shape)
    print(mesh_mu.repeat(batch_size,1,1).view(batch_size, -1).shape)
    print('ok')
    '''
    v_shaped =  (
        torch.matmul(
            betas.unsqueeze(1),
            mesh_pca.repeat(batch_size,1,1,1).permute(0,3,1,2).contiguous().view(batch_size,bases_num,-1)
        ).squeeze(1)
        + mesh_mu.repeat(batch_size,1,1).view(batch_size, -1)
    ).view(batch_size, mesh_num, 3)

    pose_weights = get_poseweights(poses, batch_size)

    #print(posedirs.repeat(batch_size,1,1,1).shape)
    #print((pose_weights.view(batch_size,1,(keypoints_num - 1)*9,1)).repeat(1,mesh_num,1,1).shape)
    #input()

    v_posed =    \
        v_shaped + \
        torch.matmul(
            posedirs.repeat(batch_size, 1, 1, 1),
            (pose_weights.view(
                batch_size,
                1,
                (keypoints_num - 1) * 9,
                1,
            )).repeat(1, mesh_num, 1, 1)
        ).squeeze(3)

    J_posed = torch.matmul(v_shaped.permute(0,2,1),J_regressor.repeat(batch_size,1,1).permute(0,2,1))
    J_posed = J_posed.permute(0, 2, 1)
    J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]

    pose = poses.permute(1, 0, 2)
    pose_split = torch.split(pose, 1, 0)


    angle_matrix=[]
    for i in range(keypoints_num):
        out, tmp = rodrigues(pose_split[i].contiguous().view(-1, 3))
        angle_matrix.append(out)

    #with_zeros = lambda x: torch.cat((x,torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1)),1)

    with_zeros = lambda x:\
        torch.cat((x,   Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1).cuda())  ),1)

    pack = lambda x: torch.cat((Variable(torch.zeros(batch_size,4,3).cuda()),x),2)

    results = {}
    results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size,3,1)),2))

    for i in range(1, kintree_table.shape[1]):
        tmp = with_zeros(torch.cat((angle_matrix[i],
                         (J_posed_split[i] - J_posed_split[parent[i]]).view(batch_size,3,1)),2))
        results[i] = torch.matmul(results[parent[i]], tmp)

    results_global = results

    results2 = []

    for i in range(len(results)):
        vec = (torch.cat((J_posed_split[i], Variable(torch.zeros(batch_size,1).cuda()) ),1)).view(batch_size,4,1)
        results2.append((results[i]-pack(torch.matmul(results[i], vec))).unsqueeze(0))

    results = torch.cat(results2, 0)

    T = torch.matmul(results.permute(1,2,3,0), weights.repeat(batch_size,1,1).permute(0,2,1).unsqueeze(1).repeat(1,4,1,1))
    Ts = torch.split(T, 1, 2)
    rest_shape_h = torch.cat((v_posed, Variable(torch.ones(batch_size,mesh_num,1).cuda()) ), 2)
    rest_shape_hs = torch.split(rest_shape_h, 1, 2)

    v = Ts[0].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1, mesh_num)\
        + Ts[1].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1, mesh_num)\
        + Ts[2].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1, mesh_num)\
        + Ts[3].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1, mesh_num)

    #v = v.permute(0,2,1)[:,:,:3]
    Rots = rodrigues(rots)[0]


    Jtr = []



    for j_id in range(len(results_global)):
        #print(results_global[j_id][:,:3,3:4].shape)
        #input()
        Jtr.append(results_global[j_id][:,:3,3:4])
    '''
    Jtr.insert(4,v[:,:3,333].unsqueeze(2))
    Jtr.insert(8,v[:,:3,444].unsqueeze(2))
    Jtr.insert(12,v[:,:3,672].unsqueeze(2))
    Jtr.insert(16,v[:,:3,555].unsqueeze(2))
    Jtr.insert(20,v[:,:3,745].unsqueeze(2))
    '''
    # root joint is not wrist but center of palm
    #Jtr.insert(1,(0.5 * (v[:,:3,17] + v[:,:3,67])).unsqueeze(2))
    #Jtr[0] = (0.5 * (v[:,:3,17] + v[:,:3,67])).unsqueeze(2)

    Jtr = torch.cat(Jtr, 2) #.permute(0,2,1)

    v = torch.matmul(Rots,v[:,:3,:]).permute(0,2,1) #.contiguous().view(batch_size,-1)
    Jtr = torch.matmul(Rots,Jtr).permute(0,2,1) #.contiguous().view(batch_size,-1)

    #return torch.cat((Jtr,v), 1)
    return v

def decompose_par(parameters):  # 82

    rots = parameters[:,:3]
    poses = parameters[:,3:72]
    betas = parameters[:,72:]

    return rots, poses, betas


def main():

   
   
    rots = (torch.zeros(1,3)).cuda()    #global rotation
    poses = (torch.zeros(1,23,3) * .2).cuda()   #joints
    betas = (torch.zeros(1,10) * .03).cuda()    #

    rots[0][2]=3.1415
    
    # note tensors start from zero, so joint number has to -1
    poses[0][8]=1
    #print(poses)
    #betas[0][6]=10
    #print(betas)


    Jtr,vertices = par_to_mesh(rots, poses, betas)


    outmesh_path = 'hello_smpl.obj'

    # Write Vertices

    with open( outmesh_path, 'w') as fp:
        for v in vertices[0]:
            #fp.write( 'v %f %f %f\n' % ( float(v[0]),float(v[1]), float(v[2])) )
            fp.write( 'v %f %f %f\n' % (v[0], v[1],v[2]))

    # Write Face (f)

    file1 = open('models/SMPL_f.obj')
    content = file1.readlines()
    file1.close()

    with open( outmesh_path, 'a') as fp:
        for i,x in enumerate(content):
            a = x[:len(x)-1].split(" ")
            fp.write(x)

    ## Print message
    print '..Output mesh saved to: ', outmesh_path

if __name__ == "__main__":
    main()
