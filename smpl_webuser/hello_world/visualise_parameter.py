from render import rendering
from mesh_generator import generate_mesh
import argparse
import pickle
import numpy as np
import os
import csv
from PIL import Image
from scale_mask import scale_fixed_height
from calculate_circumference import get_measurements
from smpl_webuser.serialization import load_model


# check the model's prediction 
# '''
batch_size=1
dataset_size = 9
'''
parent_dic = "/home/yifu/workspace/data_smpl/real_human/scaled"
folder_name = 'orthographic_86'
path = os.path.join(parent_dic, folder_name)
'''
folder_name = 'FAUST_male'
path = '/home/yifu/workspace/Data/MPI-FAUST/training/registrations_obj/male/test_model'
m = load_model('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
#real_height=[168, 168, 170, 172, 160, 181, 172, 179, 183]
real_height = [176.0837, 169.6549, 177.9879, 176.6942, 178.0513]
#parent_dic = "/home/yifu/workspace/data_smpl/A_pose_3"
#path = os.path.join(parent_dic, 'test')


csv_name = os.path.join(path,'%s.csv'%folder_name)
# Check real human
with open(csv_name, mode='w') as csv_file:
        for i in range(dataset_size):
                # unpickle parameters
                pickle_path = '%s/%d'% (path, i)
                #pickle_path = '%s/set_%d_%s'% (path,n, mode)
                infile = open(pickle_path, 'rb')
                parameters = pickle.load(infile)   
                #print(parameters)
                #parameters = parameters[:79]
                if parameters.shape[0] > 82:
                        measurements = parameters[82:]
                else:
                        measurements = np.zeros(4)
                parameters = parameters[:82]
                parameters[:72] = np.zeros(72)
                parameters[2] = np.pi
                parameters[16*3 + 2]=-np.pi/4 #shoulder joint
                parameters[16*3 + 1]=-np.pi/32
                parameters[17*3 + 2]=np.pi/4  #shoulder joint
                parameters[17*3 + 1]=np.pi/32

                h, w, c, n, a = get_measurements(m, parameters)
                
                
                ratio = real_height[i]/h
                

                csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
                csv_writer.writerow([i,'Height',real_height[i]])
                csv_writer.writerow([i,'Waist',w*ratio, measurements[0]/3*real_height[i]])
                csv_writer.writerow([i,'Chest',c*ratio, measurements[1]/3*real_height[i]])
                csv_writer.writerow([i,'Neck',n*ratio, measurements[2]/3*real_height[i]])
                csv_writer.writerow([i,'Arm',a*ratio, measurements[3]/3*real_height[i]])
                csv_writer.writerow(['\n'])
                #parameters = np.append(np.zeros(3),parameters)
                #parameters[0] = np.pi
                
                
                # generate mesh
                img_name = '%d'%i
                generate_mesh(parameters,batch_size,path,img_name)
                # render silhouettes
                '''
                img_name = '%d'%(n) # set_0_predict
                rendering(parameters,batch_size,path,img_name,400,400)
                '''
                img_name = '%d_0_p' % i
                angle = np.pi
                axis = np.array([0, 1, 0])
                rendering(m, parameters, batch_size, path, img_name,
                        400, 400, angle, axis)
                '''
                img_name = '%d_0_p.png'%i
                img_name = os.path.join(path, img_name)
                image = Image.open(img_name)
                output = scale_fixed_height(image,264,192)
                img_name = '%d_0_p_s.png'%i
                img_name = os.path.join(path, img_name)
                output.save(img_name)
                '''
                

                # save side view image
                img_name = '%d_1_p' % i
                angle = np.pi/2
                axis = np.array([0, 1, 0])
                rendering(m, parameters, batch_size, path, img_name,
                        400,400, angle, axis)
                '''
                img_name = '%d_1_p.png'%i
                img_name = os.path.join(path, img_name)
                image = Image.open(img_name)
                output = scale_fixed_height(image,264,192)
                img_name = '%d_1_p_s.png'%i
                img_name = os.path.join(path, img_name)
                output.save(img_name)
                '''

csv_file.close()

'''
# normal mode
for n in range(dataset_size):
    for mode in ['predict', 'actual']:
        # unpickle parameters
        pickle_path = '%s/%d_%s'% (path, n, mode)
        #pickle_path = '%s/set_%d_%s'% (path,n, mode)
        infile = open(pickle_path, 'rb')
        parameters = pickle.load(infile)   # 72pose+10shape=82
        #print(parameters.shape)
        parameters = parameters[:79]
        #print(parameters.shape)
        print(mode)
        parameters = np.append(np.zeros(3),parameters)
        #parameters[0] = np.pi
 
        # render silhouettes
        img_name = '%d_%s'%(n,mode) # set_0_predict
        rendering(parameters,batch_size,path,img_name,400,400)
        # generate mesh
        generate_mesh(parameters,batch_size,path,img_name)
'''


'''
# generate some data to test the pose[0-2]
pose = np.zeros(72)
betas = np.zeros(10)
# pose[0] pitch 
#pose[0]=0  #upside down
pose[0]=np.pi   #frontal view

#pose[1]=3
#pose[2]=10
batch_size = 1
path = 'dataset/temp'
img_name = 'side_view'

#pose[0]-=0.65
#pose[1]+=0.5
pose[2]+=3.25

parameters = np.append(pose[:],betas[:])
rendering(parameters,batch_size,path,img_name)

'''
'''
for i in range(0,10):
    betas[i]=15
    img_name = 'betas[%d]=15'%i
    pose[0]=np.pi
    parameters = np.append(pose[:],betas[:])
    rendering(parameters,batch_size,path,img_name)

    pose[0]=0
    parameters = np.append(pose[:],betas[:])
    generate_mesh(parameters,batch_size,path,img_name)

    betas[i]=0
'''