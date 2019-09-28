import sys
sys.path.append("/home/yifu/workspace/smpl")
from smpl_webuser.serialization import load_model
import numpy as np


## Load SMPL model (here we load the male model)
m = load_model( '../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl' )


def generate_mesh(parameters, batch, path, name):
  for i in range(0,batch):
    if batch==1:
        m.pose[:] = parameters[:72]   #72pose+10shape=82
        m.betas[:] = parameters[72:]
    else:                
        m.pose[:] = parameters[i][:72]   
        m.betas[:] = parameters[i][72:]
    ## Write to an .obj file
    outmesh_path = '%s/%s_%d.obj'% (path,name,i)
    with open( outmesh_path, 'w') as fp:
        for v in m.r:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in m.f+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
    ## Print message
    print('..Output mesh saved to: ', outmesh_path) 
