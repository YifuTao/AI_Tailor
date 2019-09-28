import torch
import pickle
import numpy as np
import sys
sys.path.append('functions')
from SMPL_Pytorch import rodrigues, get_poseweights, par_to_mesh, decompose_par
from render import mesh2Image, transpose_mesh, scale_mesh
from body_measurements import vertex2measurements


gender = 'male'
m = pickle.load(open('models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % gender[0]))

#rots = (torch.zeros(1,3)).cuda()    # global rotation
#poses = (torch.zeros(1,23,3) * .2).cuda()   # joints
#betas = (torch.zeros(1,10) * .03).cuda()    # shapes

#betas[0][:]=torch.FloatTensor([0.0531, 0.1330, -0.5738, 0.4583, -0.2401, -0.1515, 0.1105, -0.0679, -0.0311, -0.1351])
par_prd = torch.FloatTensor([[0.0000e+00,  0.0000e+00,  3.1416e+00,  7.0868e-03, -1.0829e-02,
         1.3441e-01,  8.2359e-03, -8.3458e-03,  7.1836e-04,  1.8783e-02,
        -1.8686e-02, -7.1865e-03, -1.0443e-03,  2.8361e-03, -8.5106e-03,
        -1.4859e-02,  8.6934e-03,  3.0642e-03, -1.1795e-02,  1.1339e-02,
         1.6469e-02,  1.2701e-02, -9.2002e-03, -8.6515e-03,  7.2081e-03,
         1.7074e-02,  1.7403e-03, -1.2998e-02, -2.1399e-04, -4.5735e-03,
        -1.7027e-02,  1.5658e-02, -1.6932e-02,  1.4686e-02, -1.6925e-02,
         1.1620e-03,  1.8845e-02, -9.6120e-03, -1.6456e-02, -1.0575e-02,
        -2.9572e-03, -1.8069e-02,  5.3464e-03, -1.9481e-02, -5.1072e-04,
        -1.1627e-02,  1.2229e-02,  1.9346e-02, -1.7012e-02, -8.8512e-02,
        -5.8843e-01,  6.8711e-03,  8.0142e-02,  6.0888e-01,  1.9338e-02,
        -1.1163e-02,  8.0411e-03, -1.8523e-02,  9.3953e-03,  1.2386e-02,
        -1.6128e-02,  1.6342e-02,  4.3522e-03, -1.5458e-02,  1.9325e-02,
         9.3975e-03,  1.8250e-02, -1.9811e-02, -1.5826e-02, -1.4165e-02,
        -5.1610e-03,  1.1331e-02, -3.4175e+00,  8.6845e-01, -2.2580e+00,
         7.0781e-01,  1.3401e+00,  9.4737e-01,  2.7599e+00, -2.6706e+00,
        -1.8776e+00,  1.6441e+00]]).cuda()
rots, poses, betas = decompose_par(par_prd)
vertices = par_to_mesh('male', rots, poses, betas)
X, Y, Z = [vertices[:,:, 0], vertices[:,:, 1], vertices[:,:, 2]]
h, w, c, n, a = vertex2measurements(X, Y, Z)

#vertices = par_to_mesh(gender, rots, poses, betas)
        
# rendering
parent_dic = '/home/yifu/workspace/data_smpl/test'
vertices_num = 6890
path = parent_dic
batch = 1

# frontal view
name = 'test_0'
v = vertices.squeeze().to("cpu").numpy()
angle = np.pi
axis = np.array([0, 0, 1])
v = transpose_mesh(v, angle, axis)
angle = np.pi
axis = np.array([0, 1, 0])
v = transpose_mesh(v, angle, axis)
v_0 = scale_mesh(v, 400, 400, vertices_num=vertices_num)
mesh2Image(v_0, m['f'], batch, path, name, 400, 400, vertices_num=vertices_num)
# side view
name = 'test_1'
angle = np.pi/2
axis = np.array([0, -1, 0])
v = transpose_mesh(v, angle, axis)
v_1 = scale_mesh(v, 400, 400, vertices_num=vertices_num)
mesh2Image(v_1, m['f'], batch, path, name, 400, 400, vertices_num=vertices_num)


# mesh generator
outmesh_path = join(path, '%d.obj'%i)

with open(outmesh_path, 'w') as fp:
        for v in vertices[0]:
        #fp.write( 'v %f %f %f\n' % ( float(v[0]),float(v[1]), float(v[2])) )
                fp.write( 'v %f %f %f\n' % (v[0], v[1],v[2]))
        for f in m['f']+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print '..Output mesh saved to: ', outmesh_path

#X, Y, Z = [mesh[:,:, 0], mesh[:,:, 1], mesh[:,:, 2]]
#h_prd, w_prd, c_prd, n_prd, a_prd = vertex2measurements(X, Y, Z)

m = load_model('models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
#m.betas[5]=10
#m.pose[:] = par_prd[1][:72].to("cpu").detach().numpy()
#m.betas[:] = par_prd[1][72:].to("cpu").detach().numpy()
'''
Xm, Ym, Zm = [m.r[:,0],m.r[:,1],m.r[:,2]]
Xm=torch.from_numpy(np.expand_dims(Xm, axis=0)) # 1,6890
Ym=torch.from_numpy(np.expand_dims(Ym, axis=0))
Zm=torch.from_numpy(np.expand_dims(Zm, axis=0))
hm, wm, cm, nm, am = vertex2measurements(Xm, Ym, Zm)
'''