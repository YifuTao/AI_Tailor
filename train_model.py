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

import sys
sys.path.append('functions')
from SMPL_Pytorch import par_to_mesh, decompose_par
from body_measurements import vertex2measurements

from smpl_webuser.serialization import load_model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument(
        "--num_epochs",
        default=25,
        type=int,
        help="Total number of epochs for training [25]",
    )
    parser.add_argument(
        "--batch_size",
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
        default=1,
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
        "--par_loss_weight",
        default=1,
        type=float,
        help="parameter loss weight [1]"
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
        "--iteration",
        default=1,
        type=int,
        help="number of iteration in the regression [1]"
    )
    return parser.parse_args()


def split_dataset(total_dataset_size):
    dataset_size = {
        'train': int(total_dataset_size*0.95),
        'val': int(total_dataset_size*0.05),
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
    dataloader = {x: DataLoader(silhouette[x], shuffle=True, batch_size=args.batch_size)
                  for x in ['train', 'val']}

    return dataloader


def myresnet50(device, num_output=82, use_pretrained=True, num_views=1, num_iteration=1):

    import resnet_multi_view
    model = resnet_multi_view.resnet50(
        pretrained=use_pretrained, num_views=num_views, num_iteration=num_iteration, num_output=num_output)
    #model = torchvision.models.resnet50(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features # * num_views

    # print(num_ftrs)    # 2048 : features
    # print(model.fc.out_features)   #1000 resnet deafult: number of output classes
    # change output number of classes
    model.fc = nn.Linear(num_ftrs + num_output, num_output)
    model.mlp = nn.Linear(num_ftrs + num_output, num_ftrs)

    model = model.to(device)
    # model = nn.DataParallel(model)     # multi GPU
    return model


def train_model(parent_dic, save_name, vis_title, device, model, dataloader, criterion, optimiser, scheduler, args,):
    import time
    import copy
    
    since = time.time()
    a = []
    while len(a) != 8:
        print('weights: pose shape ver h c w n a:')
        a = [0.1, 0.1, 0.1, 0.0, 0.01, 0.01, 0.01, 0.01]   # no height loss
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
    print('----------------------------------------------------')
    record = open(join(parent_dic, 'trained_model',save_name+'_record.txt'),'w+')
    record.write('Gender:%s\n'%args.gender)
    record.write('Dataset:%d\n'%args.dataset_size)
    record.write('\nParameter Weights\n')
    record.write('Pose %.3f Shape %.3f Ver %.3f\n'%(pose_w,shape_w,ver_w))
    record.write('Height %.3f Chest %.3f Waist %.3f Neck %.3f Arm %.3f\n'%(h_w,c_w,w_w,n_w,a_w))
    record.write('\n')
    record.close()

    vis = visdom.Visdom()
    win_shape = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title = 'shape %.2f'%shape_w,
        legend=['train_shape', 'val_shape',]))
    win_pose_ver = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title = 'pose %.2f vertices %.2f'%(pose_w,ver_w) ,
        legend=['train_pose', 'val_pose','train_ver','val_ver']))
    win_cw = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title = 'Chest %.2f Waist %.2f in cm'%(c_w,w_w) ,
        legend=['train_c','val_c','train_w','val_w']))
    win_na = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title = 'Neck %.2f Arm %.2f in cm'%(n_w,a_w) ,
        legend=['train_n','val_n','train_a','val_a']))
    

    num_epochs = args.num_epochs
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        record = open(join(parent_dic, 'trained_model',save_name+'_record.txt'),'a')
        record.write('\nEpoch {}/{}\n'.format(epoch, num_epochs - 1))
        record.write('-' * 10+'\n')
        record.close()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_pose = 0.0
            running_loss_shape = 0.0
            running_loss_ver = 0.0
            running_loss_h = 0.0
            running_loss_c = 0.0
            running_loss_w = 0.0
            running_loss_n = 0.0
            running_loss_a = 0.0

            # Iterate over data.
            for index, imgs, par_gt in dataloader[phase]:
                
                batch = par_gt.shape[0]
                inputs = torch.FloatTensor([])
                for k in range(0, len(imgs)):    # go through the views
                    inputs = torch.cat((inputs, imgs[k]), 0)
                inputs = inputs.to(device)

                par_gt = par_gt.float()  # from double to float
                par_gt = par_gt[:,:args.num_output]
                par_gt = par_gt.to(device)

                # zero the parameter gradients
                optimiser.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # prediction
                    par_prd = model(inputs)
                    # par_prd = par_prd[:,:,args.iteration-1].reshape(batch, 82)
                    rots, poses, betas = decompose_par(par_prd)
                    mesh_prd = par_to_mesh(args.gender, rots, poses, betas)

                    X, Y, Z = [mesh_prd[:,:, 0], mesh_prd[:,:, 1], mesh_prd[:,:, 2]]
                    h_prd, w_prd, c_prd, n_prd, a_prd = vertex2measurements(X, Y, Z)

                    vertices_prd = torch.reshape(mesh_prd, (batch, -1))


                    # ground truth
                    rots, poses, betas = decompose_par(par_gt)
                    mesh_gt = par_to_mesh(args.gender, rots, poses, betas)

                    X, Y, Z = [mesh_gt[:,:, 0], mesh_gt[:,:, 1], mesh_gt[:,:, 2]]
                    h_gt, w_gt, c_gt, n_gt, a_gt = vertex2measurements(X, Y, Z)
                    
                    vertices_gt = torch.reshape(mesh_gt, (batch, -1))

                    pose_loss = criterion(par_prd[:, :72], par_gt[:, :72])
                    shape_loss = criterion(par_prd[:, 72:], par_gt[:, 72:])
                    ver_loss = criterion(vertices_prd, vertices_gt)
                    
                    h_loss = criterion(h_prd, h_gt)
                    # ratio_prd = 1.76/h_prd
                    # ratio_gt = 1.76/h_gt
                    c_loss = criterion(c_prd, c_gt)
                    w_loss = criterion(w_prd, w_gt)
                    n_loss = criterion(n_prd, n_gt)
                    a_loss = criterion(a_prd, a_gt)
                    

                    loss = pose_loss * pose_w + shape_loss * shape_w + ver_loss * ver_w 
                    loss += h_loss*h_w + c_loss * c_w + w_loss *w_w + n_loss*n_w + a_loss*a_w 
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
                

            if phase == 'train':
                scheduler.step()

            dataset_size = split_dataset(args.dataset_size)
            epoch_loss = running_loss / dataset_size[phase]
            epoch_loss_pose = np.sqrt(running_loss_pose / dataset_size[phase])
            epoch_loss_shape = np.sqrt(running_loss_shape / dataset_size[phase])
            epoch_loss_ver = np.sqrt(running_loss_ver / dataset_size[phase])
            
            epoch_loss_c = 100 * np.sqrt(running_loss_c / dataset_size[phase])
            epoch_loss_w = 100 * np.sqrt(running_loss_w / dataset_size[phase])
            epoch_loss_n = 100 * np.sqrt(running_loss_n / dataset_size[phase])
            epoch_loss_a = 100 * np.sqrt(running_loss_a / dataset_size[phase])
            epoch_loss_h = 100 * np.sqrt(running_loss_h / dataset_size[phase])
            

            print('{} Loss: {:.4f} RMS Shape {:.4f} Pose {:.4F} Ver {:.4f} Chest {:.2f}cm Waist {:.2f}cm Neck {:.2f}cm Arm{:.2f}cm H{:.2f}cm'.format(
                phase[0], epoch_loss, epoch_loss_shape, epoch_loss_pose, epoch_loss_ver, epoch_loss_c, epoch_loss_w, epoch_loss_n, epoch_loss_a,epoch_loss_h))
            #record.write("Hello \n") 
            record = open(join(parent_dic, 'trained_model',save_name+'_record.txt'),'a')
            record.writelines(
                '{} Loss: {:.4f} RMS Shape {:.4f} Pose {:.4F} Ver {:.4f} Chest {:.2f}cm Waist {:.2f}cm Neck {:.2f}cm Arm{:.2f}cm H{:.2f}cm\n'.format(
                phase[0], epoch_loss, epoch_loss_shape, epoch_loss_pose, epoch_loss_ver, epoch_loss_c, epoch_loss_w, epoch_loss_n, epoch_loss_a,epoch_loss_h)
            )
            record.close()
            
            vis.line(
                X=np.array([epoch]),
                Y=np.array([epoch_loss_shape]),
                name=phase+'_shape',
                win=win_shape,
                update='append'
            )
            vis.line(
                X=np.array([epoch]),
                Y=np.array([epoch_loss_pose]),
                name=phase+'_pose',
                win=win_pose_ver,
                update='append'
            )
            vis.line(
                X=np.array([epoch]),
                Y=np.array([epoch_loss_ver]),
                name=phase+'_ver',
                win=win_pose_ver,
                update='append'
            )
            vis.line(
                X=np.array([epoch]),
                Y=np.array([epoch_loss_c]),
                name=phase+'_c',
                win=win_cw,
                update='append'
            )
            vis.line(
                X=np.array([epoch]),
                Y=np.array([epoch_loss_w]),
                name=phase+'_w',
                win=win_cw,
                update='append'
            )
            vis.line(
                X=np.array([epoch]),
                Y=np.array([epoch_loss_n]),
                name=phase+'_n',
                win=win_na,
                update='append'
            )
            vis.line(
                X=np.array([epoch]),
                Y=np.array([epoch_loss_a]),
                name=phase+'_a',
                win=win_na,
                update='append'
            )
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, join(parent_dic,'trained_model', save_name+'.pth'))
            if (epoch+1) % 5 == 0:
                torch.save(best_model_wts, join(parent_dic,'trained_model', save_name+'_epoch_%d.pth'%epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    args = parse_args()

    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print('-----------------------------------------------------------')
    print('Gender: ', args.gender)
    print('Dataset size: ', args.dataset_size)
    print('Batch size: ', args.batch_size)
    parent_dic = "/home/yifu/workspace/data/synthetic/noise_free"
    # parent_dic = raw_input('Data Path:')
    while os.path.exists(parent_dic)==False:
        print('Wrong data path!')
        parent_dic = raw_input('Data Path:')  
    if os.path.exists(join(parent_dic,'trained_model'))==False:
        print('No trained_model folder in Data path!')
        exit()
    save_name = raw_input('Name of the model weights saved:')
    # save_name = 'test'
    save_path = os.path.join(parent_dic,'trained_model', save_name+'.pth')
    
    while os.path.exists(save_path) and save_name!='test':
        print('Network weights save path will overwrite existing pth file!')
        save_name = raw_input('Name of the model weights saved:')
        save_path = os.path.join(parent_dic,'trained_model', save_name+'.pth')

    print('-----------------------------------------------------------')
    print('Save path: ', save_path)

    print('-----------------------------------------------------------')
    if raw_input('Confirm the above setting? (yes/no): ')!='yes':
        print('Terminated')
        exit()
    print('Training starts')
    print('-----------------------------------------------------------')

    dataloader = load_data(args.dataset_size, parent_dic, args)

    # iteration = int(raw_input('Number of iterations in the neuron network: '))
    model = myresnet50(device, num_output=args.num_output,
                       use_pretrained=True, num_views=args.num_views, num_iteration = args.iteration)
    criterion = nn.MSELoss()    # Mean suqared error for each element
    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)

    vis_title = save_name
    model = train_model(parent_dic, save_name, vis_title, device, model, dataloader, criterion,
                        optimiser, exp_lr_scheduler, args)


if __name__ == "__main__":
    main()
