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

def save_update_images(index, path, count,inputs, reprojection,gt,new_prd, batch,num_views):    # images [batch, height, width]

    tmp = count
    for view in range(0,num_views):
        count = tmp
        for b in range(0,batch):
            
            image = inputs.detach().cpu().numpy()[view*batch+b]*255
            image_path = join(path,'%d_%d.png'%(count,view))
            cv2.imwrite(image_path, image)
            
            image = reprojection.detach().cpu().numpy()[view*batch+b]*255
            image_path = join(path,'%d_%d_reproj.png'%(count,view))
            cv2.imwrite(image_path, image)

            count = count + 1

    count = tmp
    for b in range(0,batch):    

        pickle_path = join(path,'%d'%(count))
        pickle.dump(gt[b].detach().cpu(), open(pickle_path, 'wb'))
        infile = open(pickle_path, 'rb')
        tmp1 = pickle.load(infile)
        infile.close()

        pickle_path = join(path,'%d_reproj'%(count))
        prediction = torch.zeros(gt[b].shape[0])
        #    = new_prd[b].detach().cpu()
        prediction[:new_prd[b].shape[0]] = new_prd[b]
        pickle.dump(prediction.detach().cpu(), open(pickle_path, 'wb'))
        infile = open(pickle_path, 'rb')
        tmp2 = pickle.load(infile)
        infile.close()
        
        count = count + 1    

    return count


