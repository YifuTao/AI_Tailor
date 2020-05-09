import cv2
import numpy
import torch
from os.path import join
import pickle
from PIL import Image



def save_images(path, name, images):    # images [batch, height, width]

    for i in range(0, images.shape[0]):
        image = images.detach().cpu().numpy()[i]*255
        cv2.imwrite(join(path, name+'_%d.png'%i), image)

def save_update_images(path, count,inputs, reprojection,par, index, batch,num_views):    # images [batch, height, width]

    for view in range(0,num_views):
        for b in range(0,batch):
            '''
            image = inputs.detach().cpu().numpy()[view*batch+b]*255
            image_path = join(path,'%d_%d.png'%(index[b],view))
            cv2.imwrite(image_path, image)
            '''
            image = reprojection.detach().cpu().numpy()[view*batch+b]*255
            image_path = join(path,'%d_%d_reproj_%d.png'%(index[b],view,count))
            cv2.imwrite(image_path, image)
    for b in range(0,batch):        
        pickle_path = join(path,'%d_reproj_%d'%(index[b],count))
        pickle.dump(par[b], open(pickle_path, 'wb'))
        infile = open(pickle_path, 'rb')
        tmp = pickle.load(infile)
        infile.close()


