import pickle
from os.path import join
import numpy as np
from os.path import isfile, join
from os import listdir
import sys
sys.path.append("/home/yifu/workspace/smpl")
from smpl_webuser.serialization import load_model


gender = 'male'
m = load_model('../../models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % gender[0])
path = '/home/yifu/workspace/data_smpl/A_pose_5/male/noise_free_copy/train'
print path
files = [f for f in listdir(path) if isfile(join(path, f))]

for name in files:
    if name[-4:]=='.png':
        continue
    pickle_path = join(path, name)
    infile = open(pickle_path, 'rb')
    par = pickle.load(infile)
    if par.shape[0]>82:
        print(name, ' ', par.shape[0], ' skip')
        continue
    m.pose[:] = par[:72]   # 72pose+10shape=82
    m.betas[:] = par[72:]   #82+20670=20752
    v = np.reshape(m.r, -1)
    out = np.concatenate((par, v), axis=0)

    pickle.dump(out, open(pickle_path, 'wb'))