import cv2
import numpy
import torch
from os.path import join

def save_images(path, name, images):    # images [batch, height, width]

    for i in range(0, images.shape[0]):
        image = images.detach().cpu().numpy()[i]*255
        cv2.imwrite(join(path, name+'_%d.png'%i), image)