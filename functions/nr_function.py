import cv2
import numpy
import torch
from os.path import join

def save_images(path, name, images):    # images [batch, height, width]

    for i in range(0, images.shape[0]):
        image = images.detach().cpu().numpy()[i]*255
        cv2.imwrite(join(path, name+'_%d.png'%i), image)

def save_update_images(path, count,inputs, reprojection, parameter, batch):    # images [batch, height, width]

    for i in range(0, inputs.shape[0]):
        image = inputs.detach().cpu().numpy()[i]*255
        cv2.imwrite(join(path, '%d_0.png'%(count+i)), image)

        image = reprojection.detach().cpu().numpy()[i]*255
        cv2.imwrite(join(path, '%d_1.png'%(count+i)), image)

        num = count + i%batch
        pickle.dump(parameters[num], open('%s/%d' % (path, num), 'wb'))


