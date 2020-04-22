import torch
from torch import all, eq
import torchvision
import os
from PIL import Image
from torchvision.transforms import ToTensor, functional
import argparse
from os.path import isfile, join
from os import listdir

def parse_args():
    parser = argparse.ArgumentParser(description="generate dataset for training & validation")
    parser.add_argument(
        "--dataset_size",
        default=1000,
        type=int,
        help="size of the whole (train+val) dataset [1000]",
    )
    parser.add_argument(
        "--num_views",
        default=2,
        type=int,
        help="Number of views as input [2]",
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
    return parser.parse_args()

def img_size(image):
    img = ToTensor()(image)
    img = img[0]

    height = img.size()[0] #100   height
    width = img.size()[1] #50    wids

    zeros = torch.zeros(width)
    for k in range(0, height):
        if(all(eq(img[k][:],zeros)).numpy()==0):
            row_up = k
            break
    for k in reversed(range(0, height)):
        if(all(eq(img[k][:],zeros)).numpy()==0):
            row_down = k
            break

    img = torch.t(img)

    zeros = torch.zeros(height)
    for k in range(0, width):
        if(all(eq(img[k][:],zeros)).numpy()==0):
            col_left = k
            break
    for k in reversed(range(0, width)):
        if(all(eq(img[k][:],zeros)).numpy()==0):
            col_right = k
            break
    return row_up,row_down,col_left,col_right


def scale_preserve(image, output_height, output_width):
    
    row_up,row_down,col_left,col_right = img_size(image)

    img = ToTensor()(image)
    img = img[0]    #RGB to Binary

    img_height = row_down - row_up
    img_width = col_right - col_left
    img_ratio = img_height / img_width
    # Scale image to make the bounding box has the height of 264
    output_ratio = output_height / output_width
    
    corner_i = row_up
    corner_j = col_left

    if img_ratio > output_ratio:
        # long image, keep height and increase crop width
        crop_h = img_height
        crop_w = round(img_height / output_ratio)
        # shift image to the right to centre
        corner_j -= round((crop_w - img_width)/2)
    else:
        # wide image, increase crop height
        crop_h = round(img_width * output_ratio)
        crop_w = img_width
        # shift iamge down to centre
        corner_i -= round((crop_h - img_height)/2)

    output = functional.resized_crop(image,corner_i,corner_j,crop_h,crop_w,(output_height,output_width),interpolation=2)
    return output

def scale_fixed_height(image,output_height, output_width):

    row_up,row_down,col_left,col_right = img_size(image)

    img_height = row_down - row_up
    img_width = col_right - col_left
    # Scale image to make the bounding box has the height of 264
    output_ratio = float(output_height) / output_width  # to get true division
    if output_height == type(1):
        print('error in output ratio: not float!')
    
    crop_h = img_height
    crop_w = crop_h / output_ratio

    corner_i = row_up
    corner_j = col_left
    corner_j -= round((crop_w - img_width)/2)

    output = functional.resized_crop(image,corner_i,corner_j,crop_h,crop_w,(output_height,output_width),interpolation=2)
    return output

def scale_fixed_height_ratio(image, real_height, height_ratio, output_height, output_width):
    row_up,row_down,col_left,col_right = img_size(image)
    crop_h = row_down - row_up  # initial
    crop_w = col_right - col_left
    corner_i = row_up
    corner_j = col_left

    scaled_height = round(real_height * height_ratio)  # final img height
    scaled_width = round(scaled_height * crop_w / crop_h)

    scaled_img = functional.resized_crop(image,corner_i,corner_j,crop_h,crop_w,(scaled_height,scaled_width),interpolation=2)
    print(scaled_img.shape)

    scaled_img = ToTensor()(scaled_img)
    scaled_img = scaled_img[0]

    output = torch.empty(output_height, output_width)



    

def main():
    args = parse_args()
    parent_path = "/home/yifu/workspace/data_smpl/A_pose_5/male/noisy_original"
    path = join(parent_path,'val')
    #path = '/home/yifu/workspace/data_smpl/test'
    print('path: ',path)
    images = [f for f in listdir(path) if isfile(join(path, f))]

    for name in images:
        if name[-4:]!='.png':
            #print('not png!')
            continue
        fileName = join(path, name)
        image = Image.open(fileName)
        
        #'''
        output = scale_fixed_height(image,args.output_height,args.output_width)
        name = name[:-4]
        output_name = join(path, name+'.png')
        #output_name = joint(parent_path,'scale','train',name+'.png')
        #print(output_name)
        output.save(output_name)
        #'''



if __name__ == "__main__":
    main()
