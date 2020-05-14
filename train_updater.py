from __future__ import print_function, division
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from os.path import join
import visdom
import numpy as np
import pickle
import shutil
import glob
import csv
import numpy as np

import sys
sys.path.append('functions')
from SMPL_Pytorch import par_to_mesh, decompose_par
from body_measurements import vertex2measurements

from smpl_webuser.serialization import load_model
import neural_renderer as nr
from data_generator import scale_vert_nr_batch_forLoop
from nr_function import save_images,save_update_images
from SMPL_Pytorch import generate_obj


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument(
        "--epochs",
        default=25,
        type=int,
        help="Total number of epochs for training [25]",
    )
    parser.add_argument(
        "--batch",
        default=13,
        type=int,
        help="Batch size [10]",
    )
    parser.add_argument(
        "--normalise_scale",
        default=1,
        type=int,
        help="Factor for normalising ground truth [1]"
    )
    parser.add_argument(
        "--dataset_size",
        default=1000,
        type=int,
        help="Size of the whole (train+val) dataset [1000]",
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="Learning rate [0.01]",
    )
    parser.add_argument(
        "--num_views",
        default=2,
        type=int,
        help="Number of views as input [2]",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="which gpu to train [1]"
    )
    parser.add_argument(
        "--num_output",
        default=82,
        type=int,
        help="the number of output ground truth [82]"
    )
    parser.add_argument(
        "--gender",
        default='male',
        type=str,
        help="Gender ['male']"
    )
    parser.add_argument(
        "--save_path",
        default='trained_model',
        type=str,
        help="save name of the trained model ['trained_model']"
    )
    parser.add_argument(
        "--reprojection_loss",
        default=False,
        type=bool,
        help="whether use reprojection loss [False]"
    )
    parser.add_argument(
        "--cam_loss",
        default=0.1,
        type=float,
        help="camera loss [0.1]"
    )
    parser.add_argument(
        "--visdom",
        default=False,
        type=bool,
        help="switch on/off visdom [False]"
    )
    parser.add_argument(
        "--reprojection_loss_weight",
        default=1e-3,
        type=float,
        help="reprojection loss weight [0.001]"
    )
    parser.add_argument(
        "--gtCamera",
        default=False,
        type=bool,
        help="use ground truth camera pose [n]"
    )
    parser.add_argument(
        "--gtPose",
        default=False,
        type=bool,
        help="use ground truth human pose [n]"
    )
    parser.add_argument(
        "--iteration",
        default=1,
        type=int,
        help=" [n]"
    )
    parser.add_argument(
        "--update_period",
        default=3,
        type=int,
        help=" [n]"
    )
    parser.add_argument(
        "--improve_ratio",
        default=0.7,
        type=float,
        help=" [n]"
    )
    parser.add_argument(
        "--ssh",
        default=False,
        type=bool,
        help=" [n]"
    )
    parser.add_argument(
        "--hard",
        default=False,
        type=bool,
        help=" [n]"
    )

    return parser.parse_args()


def split_dataset(total_dataset_size):
    dataset_size = {
        'train': int(total_dataset_size*0.95),
        'val': int(total_dataset_size*0.05)+1,
    }
    return dataset_size


def load_data(total_dataset_size, parent_dic, args):
    import torchvision.transforms as transforms
    from torchvision.transforms import ToTensor, Compose, CenterCrop
    import os
    from dataset import HumanTestSet
    from torch.utils.data import Dataset, DataLoader

    # Image transform
    # img_transform = Compose([CenterCrop(400),ToTensor(),])  # for unscaled
    img_transform = ToTensor()  # for scaled

    dataset_size = split_dataset(total_dataset_size)
    directory = {x: os.path.join(parent_dic, x) for x in ['train', 'val']}
    silhouette = {x: HumanTestSet(
        directory[x], dataset_size[x], img_transform, num_views=args.num_views) for x in ['train', 'val']}
    dataloader = {x: DataLoader(silhouette[x], shuffle=True, batch_size=args.batch)
                  for x in ['train', 'val']}

    return dataloader

def load_data_updater(dataset_size, parent_dic, args):
    import torchvision.transforms as transforms
    from torchvision.transforms import ToTensor, Compose, CenterCrop
    import os
    from dataset import HumanTestSet,UpdateSet
    from torch.utils.data import Dataset, DataLoader

    # Image transform
    # img_transform = Compose([CenterCrop(400),ToTensor(),])  # for unscaled
    img_transform = ToTensor()  

    # dataset_size = split_dataset(total_dataset_size)    # TODO
    directory = {x: os.path.join(parent_dic, x) for x in ['train', 'val']}
    silhouette = {x: UpdateSet(
        directory[x], dataset_size[x], img_transform, num_views=args.num_views) for x in ['train', 'val']}
    dataloader = {x: DataLoader(silhouette[x], shuffle=False, batch_size=args.batch)
                  for x in ['train', 'val']}

    return dataloader


def predictor(device, num_output=82, use_pretrained=True, num_views=1):

    import resnet_multi_view
    model = resnet_multi_view.resnet50(
        pretrained=use_pretrained, num_views=num_views)
    #model = torchvision.models.resnet50(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features # * num_views

    # print(num_ftrs)    # 2048 : features
    # print(model.fc.out_features)   #1000 resnet deafult: number of output classes
    # change output number of classes
    model.fc = nn.Linear(num_ftrs, num_output)

    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    with torch.no_grad():
        model.conv1.weight[:] = torch.unsqueeze(weight[:,0],1)

    model = model.to(device)
    # model = nn.DataParallel(model)     # multi GPU
    return model

def updater_cam(device, num_output=1, use_pretrained=True, num_views=1):

    import resnet_6channels
    model = resnet_6channels.resnet50(
        pretrained=use_pretrained,
    )
    num_ftrs = model.fc.in_features # * num_views
    model.fc = nn.Linear(num_ftrs, num_output)
    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    with torch.no_grad():
        model.conv1.weight[:,0] = weight[:,0]
        model.conv1.weight[:,1] = weight[:,1]


    model = model.to(device)
    return model

def updater_cam_sharedW(device, num_output=1, use_pretrained=True, num_views=1):

    import resnet_3channels
    model = resnet_3channels.resnet50(
        pretrained=use_pretrained,
    )
    num_ftrs = model.fc.in_features # * num_views
    model.fc = nn.Linear(num_ftrs*2, num_output)

    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    with torch.no_grad():
        model.conv1.weight[:] = torch.unsqueeze(weight[:,0],1)

    model = model.to(device)
    return model

def relu_mse_loss(output,target):
    # loss_cmp = torch.mean((output - target)**2)
    relu = nn.ReLU()
    #tmp = output-target
    #tmp_ = relu(tmp)
    loss = torch.mean(relu(output - target)**2)
    return loss

def three_channel(imgs): # make an image three identical channel 
    imgs = imgs[:,0,:,:]
    imgs=torch.unsqueeze(imgs,1)
    imgs=torch.cat((imgs,imgs,imgs),1)
    return imgs

def visdom_init(a):
    vis = visdom.Visdom()
    win_shape = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title = 'shape %.2f'%a[1],
        legend=['train_shape', 'val_shape',]))
    win_pose_ver = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title = 'pose %.2f vertices %.2f'%(a[0],a[2]) ,
        legend=['train_pose', 'val_pose','train_ver','val_ver']))
    win_cw = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title = 'Chest %.2f Waist %.2f in cm'%(a[4],a[5]) ,
        legend=['train_c','val_c','train_w','val_w']))
    win_na = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title = 'Neck %.2f Arm %.2f in cm'%(a[6],a[7]) ,
        legend=['train_n','val_n','train_a','val_a']))
    return vis, win_shape, win_pose_ver, win_cw, win_na

def visdom_append(vis, epoch,epoch_loss_shape,epoch_loss_pose,epoch_loss_ver,epoch_loss_c,epoch_loss_w,epoch_loss_n,epoch_loss_a,phase,
                win_shape,win_pose_ver,win_cw,win_na):
    vis.line(X=np.array([epoch]),Y=np.array([epoch_loss_shape]),name=phase+'_shape',win=win_shape,update='append')
    vis.line( X=np.array([epoch]), Y=np.array([epoch_loss_pose]), name=phase+'_pose', win=win_pose_ver, update='append' )
    vis.line( X=np.array([epoch]), Y=np.array([epoch_loss_ver]), name=phase+'_ver', win=win_pose_ver, update='append' )
    vis.line( X=np.array([epoch]), Y=np.array([epoch_loss_c]), name=phase+'_c', win=win_cw, update='append' )
    vis.line( X=np.array([epoch]), Y=np.array([epoch_loss_w]), name=phase+'_w', win=win_cw, update='append' )
    vis.line( X=np.array([epoch]), Y=np.array([epoch_loss_n]), name=phase+'_n', win=win_na, update='append' )
    vis.line( X=np.array([epoch]), Y=np.array([epoch_loss_a]), name=phase+'_a', win=win_na, update='append' )
    return vis

def reprojection(cam_prd, rots, poses, betas, args, batch, f_nr,n_renderer):

    mesh_cat = torch.FloatTensor([]).cuda()
    for view in range(0, args.num_views):
        rots_view = torch.zeros(batch,3).cuda()
        rots_view[:,0] = rots_view[:,0] + rots[:,0]
        rots_view[:,2] = rots_view[:,2] + rots[:,2]
        rots_view[:,1] = rots_view[:,1] + cam_prd[:, view]
        mesh_view = par_to_mesh(args.gender, rots_view, poses, betas)
        v_nr = scale_vert_nr_batch_forLoop(mesh_view)
        v_nr = mesh_view
        mesh_cat=torch.cat((mesh_cat, v_nr), 0)

    face=f_nr.repeat(mesh_cat.shape[0],1,1)
    images = n_renderer(mesh_cat, face, mode='silhouettes') # silhouettes
    return images


def train_model(parent_dic, save_name, vis_title, device, predictor, updater_cam, criterion, optimiser, scheduler, args,):
    import time
    import copy
    
    bce_loss = nn.BCELoss()
    l2_loss = nn.MSELoss()
    since = time.time()
    checkpoint = since
    a = []
    while len(a) != 8:
        print('weights: pose shape ver h c w n a:')
        # a = [0.1, 0.1, 0.1, 0.0, 0.01, 0.01, 0.01, 0.01]  # standard
        a = [0.1, 0.1, 0, 0, 0, 0, 0, 0]
        # a = [float(x) for x in raw_input().split()]
    pose_w = a[0]
    shape_w= a[1]
    ver_w=a[2]
    h_w=a[3]
    c_w=a[4]
    w_w=a[5]
    n_w=a[6]
    a_w=a[7]
    print('Parameter Weights')
    print('Pose %.3f Shape %.3f Ver %.3f'%(pose_w,shape_w,ver_w))
    print('Height %.3f Chest %.3f Waist %.3f Neck %.3f Arm %.3f'%(h_w,c_w,w_w,n_w,a_w))
    print('Camera %.3f'%args.cam_loss)
    
    print('Reprojection %.4f'%args.reprojection_loss_weight)
    print('----------------------------------------------------')
    record = open(join(parent_dic, 'trained_model',save_name+'_record.txt'),'w+')
    record.write('Gender:%s\n'%args.gender)
    record.write('Dataset:%d\n'%args.dataset_size)
    record.write('\nParameter Weights\n')
    record.write('Pose %.3f Shape %.3f Ver %.3f\n'%(pose_w,shape_w,ver_w))
    record.write('Height %.3f Chest %.3f Waist %.3f Neck %.3f Arm %.3f\n'%(h_w,c_w,w_w,n_w,a_w))
    record.write('\n')
    record.close()

    if args.visdom == True:
        vis, win_shape, win_pose_ver, win_cw, win_na = visdom_init(a)

    csv_name = join(parent_dic, 'trained_model',save_name+'.csv')
    csv_file = open(csv_name, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
    # csv_writer.writerow([' ','loss','loss']+[' ','shape','shape'])
    # csv_writer.writerow([' ','train','val']*2)
    csv_save ={
        'train': np.zeros(20),
        'val': np.zeros(20),
    }

    num_epochs = args.epochs
    best_model_wts = copy.deepcopy(predictor.state_dict())
    best_loss = float("inf")
    n_renderer = nr.Renderer(image_size=300, perspective=False, camera_mode='look_at')
    m = pickle.load(open('models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % args.gender[0]))
    f_nr = torch.from_numpy(m['f'].astype(int)).cuda()
    f_nr = f_nr[None, :, :]
    reproj_path = join(parent_dic,'trained_model','%s_reproj'%save_name)
    if os.path.exists(reproj_path):
        shutil.rmtree(reproj_path)
    os.mkdir(reproj_path)
    os.mkdir(join(reproj_path,'train'))
    os.mkdir(join(reproj_path,'val'))
    weights_path = join(parent_dic,'trained_model','%s_weights'%save_name)
    if os.path.exists(weights_path):
        shutil.rmtree(weights_path)
    os.mkdir(weights_path)
    reproj_train_path = join(parent_dic,'reprojection_%d'%args.gpu)
    if os.path.exists(reproj_train_path):
        shutil.rmtree(reproj_train_path)
    os.mkdir(reproj_train_path)
    shutil.copytree(join(parent_dic,'train'),join(reproj_train_path,'train'))
    shutil.copytree(join(parent_dic,'val'),join(reproj_train_path,'val'))
    
    for phase in ['train','val']:
        for f in glob.glob(join(reproj_train_path,phase, "*")):
            if f[-4:]!='.png':
                des = f+'_reproj'
            else:
                des = f[:-4]+'_reproj.png'
            shutil.copy(f,des)
    


    

    dataset_size = split_dataset(args.dataset_size)
    # dataset_size_total = dataset_size['train']+dataset_size['val']
    for epoch in range(num_epochs):
        count = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        record = open(join(parent_dic, 'trained_model',save_name+'_record.txt'),'a')
        record.write('\nEpoch {}/{}\n'.format(epoch, num_epochs - 1))
        record.write('-' * 10+'\n')
        record.close()

        visualise_flag = 1
        every_epoch = args.update_period
        if epoch == 0:
            reproj_round = 0
        else:
            reproj_round = int((epoch-1)/every_epoch)+1
        
        # dataset_size_total = dataset_size['train']+dataset_size['val']
        # dataset_size_total = args.dataset_size+1+epoch
         
        dataloader = load_data_updater(dataset_size, reproj_train_path, args)

        # Each epoch has a training and validation phase
        predictor.eval()
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                updater_cam.train()
            else:
                updater_cam.eval()
            
            visualise_flag = 1

            running_loss = 0.0
            running_loss_pose = 0.0
            running_loss_shape = 0.0
            running_loss_ver = 0.0
            running_loss_h = 0.0
            running_loss_c = 0.0
            running_loss_w = 0.0
            running_loss_n = 0.0
            running_loss_a = 0.0
            running_loss_cam = 0.0
            running_loss_reproj = 0.0
            running_loss_cam_delta = 0
            running_loss_cam_delta_abs = 0
            running_loss_cam_delta_input = 0

            running_loss_cam_delta_test = torch.zeros(reproj_round)
            running_loss_reproj_delta = 0
            
            # running_loss_cam_delta = torch.zeros(args.iteration)


            # Iterate over data.
            for index, imgs,reproj, gt, prm_update in dataloader[phase]:
                
                batch = gt.shape[0]
                inputs = torch.FloatTensor([])
                input_reproj = torch.FloatTensor([])
                for k in range(0, len(imgs)):    # go through the views
                    # imgs[k]=three_channel(imgs[k])
                    inputs = torch.cat((inputs, imgs[k]), 0)
                    input_reproj = torch.cat((input_reproj,reproj[k]), 0)

                inputs = inputs.to(device)
                input_reproj = input_reproj.to(device)
                prm_update = prm_update[0].to(device)

                gt = gt.float()  # from double to float
                # par_gt = par_gt[:,:args.num_output]
                gt = gt.to(device)

                # zero the parameter gradients
                optimiser.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # torch.autograd.set_detect_anomaly(True)
                    # prediction
                    prediction = predictor(inputs)
                    
                    par_prd = prediction[:,:82]
                    # use gt pose and camera
                    if args.gtPose == True:
                        par_gt_tmp = gt[:,:82]
                        par_prd[:,:72] = par_gt_tmp[:,:72]  # replace prd pose with gt pose
                    rots_prd, poses_prd, betas_prd = decompose_par(par_prd)

                    cam_prd = prediction[:,82:82+args.num_views]
                    if args.gtCamera == True:
                        cam_gt_tmp = gt[:,82:82+args.num_views]
                        cam_prd = cam_gt_tmp
                    
                    mesh_prd = par_to_mesh(args.gender, rots_prd, poses_prd, betas_prd)

                    X, Y, Z = [mesh_prd[:,:, 0], mesh_prd[:,:, 1], mesh_prd[:,:, 2]]
                    h_prd, w_prd, c_prd, n_prd, a_prd = vertex2measurements(X, Y, Z)

                    vertices_prd = torch.reshape(mesh_prd, (batch, -1))

      
                    # ground truth
                    par_gt = gt[:,:82]
                    cam_gt = gt[:,82:82+args.num_views]
                    rots_gt, poses_gt, betas_gt = decompose_par(par_gt)
                    mesh_gt = par_to_mesh(args.gender, rots_gt, poses_gt, betas_gt)

                    X_gt, Y_gt, Z_gt = [mesh_gt[:,:, 0], mesh_gt[:,:, 1], mesh_gt[:,:, 2]]
                    h_gt, w_gt, c_gt, n_gt, a_gt = vertex2measurements(X_gt, Y_gt, Z_gt)
                    
                    vertices_gt = torch.reshape(mesh_gt, (batch, -1))

                    # Silhouette Reprojection
                                        
                    reprojections = reprojection(cam_prd, rots_prd, poses_prd, betas_prd, args, batch, f_nr,n_renderer)
                    # sil_loss = bce_loss(images, inputs[:,0,:,:],)
                    sil_loss = l2_loss(reprojections, inputs[:,0,:,:],)
                    if visualise_flag == 1:
                        save_images(join(reproj_path,phase),'epoch %d,prd'%epoch,reprojections)
                        save_images(join(reproj_path,phase),'epoch %d,gt'%epoch,inputs[:,0,:,:])
                        visualise_flag = 0
                    

                    
                    loss = 0
                    
                    #print(index)
                    #save_images('/home/yifu/Data/silhouette/update/test','input',inputs[:,0,:,:])
                    # if reproj_round!=0:
                        # save_images('/home/yifu/Data/silhouette/update/test','reproj',input_reproj[:,0,:,:])
                        # input_reproj_ = torch.chunk(input_reproj,reproj_round,0)
                    
                    new_pred = prediction.clone().detach()
                    
                    # save_images('/home/yifu/Data/silhouette/update/test','reproj_%d'%r,input_reproj_[r][:,0,:,:])
                    # reprojections = torch.unsqueeze(input_reproj[:],1)
                    
                    par_update = prm_update[:,:82]
                    cam_update = prm_update[:,82:82+args.num_views]
                    rots_update, poses_update, betas_update = decompose_par(par_update)
                    # reprojection_tmp = reprojection(cam_update, rots_update, poses_update, betas_update, args, batch, f_nr,n_renderer)
                    #save_images('/home/yifu/Data/silhouette/test','reproj_%d'%r,input_reproj_[r][:,0,:,:])
                    #save_images('/home/yifu/Data/silhouette/test','reproj_prm_%d'%r,reprojection_tmp)

                    update_path = join(reproj_train_path,phase)
                    if epoch==0:
                        dataset_size[phase] = save_update_images(index,update_path, dataset_size[phase],inputs[:,0,:,:], reprojections,gt, new_pred,batch,args.num_views)
                    
                    cat_input = torch.cat((inputs,input_reproj),0)
                    cam_delta = updater_cam(cat_input)
                    cam_delta= torch.reshape(cam_delta,(args.num_views,batch))
                    cam_delta = torch.t(cam_delta)
                    new_cam = cam_delta+cam_update
                    new_pred[:,82:82+args.num_views] = new_cam

                    with torch.no_grad():
                        reprojections = reprojection(new_cam, rots_prd, poses_prd, betas_prd, args, batch, f_nr,n_renderer)
                
                    cam_delta_loss_abs = criterion (new_cam,cam_gt)
                    lambda_ = args.improve_ratio
                    # cam_delta_loss[r] = criterion(torch.abs(new_cam-cam_gt), lambda_*torch.abs(cam_update-cam_gt))
                    cam_delta_loss = relu_mse_loss(torch.abs(new_cam-cam_gt), lambda_*torch.abs(cam_update-cam_gt))
                    
                    reproj_delta_loss = criterion (reprojections, inputs[:,0,:,:])
                    dataset_size_init = split_dataset(args.dataset_size)
                    '''
                    if index[batch-1]<dataset_size_init[phase]:
                        cam_delta_loss = 0 * cam_delta_loss
                        cam_delta_loss_abs = 0 * cam_delta_loss_abs
                    '''
                    loss = loss + cam_delta_loss*args.cam_loss
                    # loss = loss + reproj_delta_loss[r]*0.001

                    # save update
                    if epoch%every_epoch == 0 and epoch>0:
                        dataset_size[phase] = save_update_images(index,update_path, dataset_size[phase],inputs[:,0,:,:], reprojections,gt, new_pred,batch,args.num_views)
                    
       
                    
                    
                    # Evaluation
                    with torch.no_grad():
                        reprojections = torch.unsqueeze(reprojection(cam_prd, rots_prd, poses_prd, betas_prd, args, batch, f_nr,n_renderer),1)
                        cam_current = cam_prd.clone().detach()
                        cam_delta_loss_test = torch.zeros(reproj_round).to(device)
                        for r in range(0,reproj_round):
                            cat_input = torch.cat((inputs,reprojections.clone().detach()),0)
                            cam_delta = updater_cam(cat_input)
                            cam_delta= torch.reshape(cam_delta,(args.num_views,batch))
                            cam_delta = torch.t(cam_delta)
                            # new_cam = cam_delta+cam_prd
                            cam_current = cam_delta+cam_current # check
                            cam_delta_loss_test[r] = criterion (cam_current,cam_gt)
                            reprojections = torch.unsqueeze(reprojection(cam_current, rots_prd, poses_prd, betas_prd, args, batch, f_nr,n_renderer),1)
                    
                    
                    
                    # Loss ------------------------------------------------
                    pose_loss = criterion(par_prd[:, :72], par_gt[:, :72])
                    shape_loss = criterion(par_prd[:, 72:], par_gt[:, 72:])
                    cam_loss = criterion(cam_prd,cam_gt)
                    ver_loss = criterion(vertices_prd, vertices_gt)
                    
                    h_loss = criterion(h_prd, h_gt)
                    # ratio_prd = 1.76/h_prd
                    # ratio_gt = 1.76/h_gt
                    c_loss = criterion(c_prd, c_gt)
                    w_loss = criterion(w_prd, w_gt)
                    n_loss = criterion(n_prd, n_gt)
                    a_loss = criterion(a_prd, a_gt)
                    
                    # loss = pose_loss * pose_w + shape_loss * shape_w + ver_loss * ver_w 
                    # loss = loss + h_loss*h_w + c_loss * c_w + w_loss *w_w + n_loss*n_w + a_loss*a_w
                    # loss = loss + cam_loss * args.cam_loss
                    # if args.reprojection_loss == True:
                    #    loss = loss + sil_loss*args.reprojection_loss_weight
                    


                  
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

                # statistics
                running_loss += loss.item() * batch
                running_loss_pose += pose_loss.item() * batch
                running_loss_shape += shape_loss.item() * batch
                running_loss_ver += ver_loss.item() * batch
                
                running_loss_h += h_loss.item() * batch
                running_loss_c += c_loss.item() * batch
                running_loss_w += w_loss.item() * batch
                running_loss_n += n_loss.item() * batch
                running_loss_a += a_loss.item() * batch
                running_loss_cam += cam_loss.item() * batch
                running_loss_reproj += sil_loss.item() * batch
                
                running_loss_cam_delta += cam_delta_loss.item() * batch
                running_loss_cam_delta_abs += cam_delta_loss_abs.item() * batch

                for i in range(0,reproj_round):
                    running_loss_cam_delta_test[i] += cam_delta_loss_test[i].item() * batch

                running_loss_reproj_delta += reproj_delta_loss.item() * batch


            if epoch%every_epoch == 0 and phase=='val':
                print(dataset_size)
            #if epoch == 0:
            #    continue
            if phase == 'train' :
                scheduler.step()

            # dataset_size = split_dataset(args.dataset_size)
            epoch_loss = running_loss / dataset_size[phase]
            epoch_loss_pose = np.sqrt(running_loss_pose / dataset_size[phase])
            epoch_loss_shape = np.sqrt(running_loss_shape / dataset_size[phase])
            epoch_loss_ver = np.sqrt(running_loss_ver / dataset_size[phase])
            epoch_loss_cam = 180/np.pi * np.sqrt(running_loss_cam / dataset_size[phase])
            epoch_loss_reproj = np.sqrt(running_loss_reproj / dataset_size[phase])

            
            epoch_loss_cam_delta = 180/np.pi * np.sqrt(running_loss_cam_delta / dataset_size[phase])
            epoch_loss_cam_delta_abs = 180/np.pi * np.sqrt(running_loss_cam_delta_abs / dataset_size[phase])
            epoch_loss_cam_delta_test = torch.zeros(reproj_round)
            for i in range(0,reproj_round):
                epoch_loss_cam_delta_test[i] = 180/np.pi * np.sqrt(running_loss_cam_delta_test[i] / dataset_size[phase])
            epoch_loss_reproj_delta = np.sqrt(running_loss_reproj_delta / dataset_size[phase])
            



            # epoch_loss_cam_delta[0] = 180/np.pi * np.sqrt(running_loss_cam_delta[0] / dataset_size[phase])
            # epoch_loss_cam_delta[1] = 180/np.pi * np.sqrt(running_loss_cam_delta[1] / dataset_size[phase])

            

            print ('cam loss %.4f'%epoch_loss_cam)
            print ('relu cam loss  %.4f'%(epoch_loss_cam_delta))
            print ('abs  cam loss  %.4f'%(epoch_loss_cam_delta_abs))
            print()
            for i in range(0,reproj_round):
                print ('test cam loss after %d update %.4f'%(i+1,epoch_loss_cam_delta_test[i]))

            print()
            '''
            print('rep loss %.4f'%epoch_loss_reproj)
            for i in range(0,reproj_round):
                print ('rep loss after %d update %.4f'%(i+1,epoch_loss_reproj_delta[i]))
            '''

            '''
            for iteration in range(0, args.iteration):
                epoch_loss_cam_delta = 180/np.pi * np.sqrt(running_loss_cam_delta[iteration] / dataset_size[phase])
                print('iteration %d cam loss %.4f'%(iteration, epoch_loss_cam_delta))
            '''

            epoch_loss_c = 100 * np.sqrt(running_loss_c / dataset_size[phase])
            epoch_loss_w = 100 * np.sqrt(running_loss_w / dataset_size[phase])
            epoch_loss_n = 100 * np.sqrt(running_loss_n / dataset_size[phase])
            epoch_loss_a = 100 * np.sqrt(running_loss_a / dataset_size[phase])
            epoch_loss_h = 100 * np.sqrt(running_loss_h / dataset_size[phase])



            print('{} Loss: {:.7f} RMS Shape {:.4f} Pose {:.4F} Ver {:.4f} Chest {:.2f}cm Waist {:.2f}cm Height{:.2f}cm Camera {:.2f}degree Reprojction {:.2f}'.format(
                phase, epoch_loss, epoch_loss_shape, epoch_loss_pose, epoch_loss_ver, epoch_loss_c, epoch_loss_w,epoch_loss_h, epoch_loss_cam,epoch_loss_reproj))
            record = open(join(parent_dic, 'trained_model',save_name+'_record.txt'),'a')
            record.writelines(
                '{} Loss: {:.7f} RMS Shape {:.4f} Pose {:.4F} Ver {:.4f} Chest {:.2f}cm Waist {:.2f}cm Neck {:.2f}cm Arm {:.2f}cm Height {:.2f}cm Camera {:.2f}degree Reprojction {:.2f}\n'.format(
                phase, epoch_loss, epoch_loss_shape, epoch_loss_pose, epoch_loss_ver, epoch_loss_c, epoch_loss_w, epoch_loss_n, epoch_loss_a,epoch_loss_h,epoch_loss_cam, epoch_loss_reproj)
            )
            for i in range(0,reproj_round):
                record.writelines('test cam loss after %d update %.4f'%(i+1,epoch_loss_cam_delta_test[i]))

            time_elapsed = time.time() - checkpoint
            checkpoint = time.time()
            print('Computation Time:{:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
            record.writelines('Computation Time:{:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
            record.close()

            if args.visdom == True:
                vis=visdom_append(vis, epoch,epoch_loss_shape,epoch_loss_pose,epoch_loss_ver,epoch_loss_c,epoch_loss_w,epoch_loss_n,epoch_loss_a,phase,
                                    win_shape,win_pose_ver,win_cw,win_na)

            for i in range(0,reproj_round):
                csv_save[phase][i] = epoch_loss_cam_delta_test[i] 
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(predictor.state_dict())
            torch.save(best_model_wts, join(weights_path, save_name+'.pth'))
            if (epoch+1) % 5 == 0:
                torch.save(best_model_wts, join(weights_path, save_name+'_epoch_%d.pth'%epoch))

        print()
        csv_writer.writerow([epoch,csv_save['train'][:reproj_round],' ',csv_save['val'][:reproj_round]])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    record = open(join(parent_dic, 'trained_model',save_name+'_record.txt'),'a')
    record.writelines('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    record.close()
    csv_file.close()

    # load best model weights
    predictor.load_state_dict(best_model_wts, strict=False)
    return predictor


def main():
    args = parse_args()

    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print('-----------------------------------------------------------')
    if args.reprojection_loss ==False:
        args.reprojection_loss_weight = 0
    print(args)
    print('Gender: ', args.gender)
    print('Dataset size: ', args.dataset_size)
    print('Batch size: ', args.batch)
    
    if args.ssh==True:
        parent_dic ='/scratch/local/ssd/yifu/Data/silhouette'
        if args.hard==True:
            parent_dic ='/scratch/local/ssd/yifu/Data/2views'
    else:
        parent_dic = "/home/yifu/Data/silhouette/small"
        if args.hard==True:
            parent_dic = "/home/yifu/Data/2views"


    # parent_dic = raw_input('Data Path:')
    
    while os.path.exists(parent_dic)==False:
        print('Wrong data path!')
        parent_dic = raw_input('Data Path:')  
    if os.path.exists(join(parent_dic,'trained_model'))==False:
        os.mkdir(join(parent_dic,'trained_model'))
    save_name = 'tmp'
    #save_name = raw_input('Name of the model weights saved:')
    weights_path = join(parent_dic,'trained_model','%s_weights'%save_name, save_name+'.pth')
    
    while os.path.exists(weights_path) and save_name!='tmp':
        print('Network weights save path will overwrite existing pth file!')
        save_name = raw_input('Name of the model weights saved:')
        weights_path = join(parent_dic,'trained_model','%s_weights'%save_name, save_name+'.pth')

    print('-----------------------------------------------------------')
    print('Network weights save path: ', weights_path)

    print('-----------------------------------------------------------')
    #if raw_input('Confirm the above setting? (y/n): ')!='y':
    #    print('Terminated')
    #    exit()
    print('Training starts')
    print('-----------------------------------------------------------')
    

    #dataloader = load_data(args.dataset_size, parent_dic, args)
    # data_path = '/home/yifu/Data/silhouette'
    # data_path = '/scratch/local/ssd/yifu/Data/silhouette/'
    
    # iteration = int(raw_input('Number of iterations in the neuron network: '))
    predictor_ = predictor(device, num_output=args.num_output,
                       use_pretrained=True, num_views=args.num_views,)
    save_path = raw_input('predictor model save path:')
    # save_path = '/home/yifu/Data/silhouette/tmp.pth'
    # save_path = join(parent_dic,'tmp.pth')
    # save_path = '/scratch/local/ssd/yifu/Data/silhouette/tmp.pth'
    predictor_.load_state_dict(torch.load(save_path))
    updater_cam_ = updater_cam_sharedW(device, num_output=1, use_pretrained=True, num_views=args.num_views)
    criterion = nn.MSELoss()    # Mean suqared error for each element
    optimiser = optim.SGD(updater_cam_.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.3)

    vis_title = save_name
    model = train_model(parent_dic, save_name, vis_title, device, predictor_, updater_cam_, criterion,
                        optimiser, exp_lr_scheduler, args)


if __name__ == "__main__":
    main()
