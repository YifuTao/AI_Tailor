from __future__ import print_function, division
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import visdom
import numpy as np


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
        help="the number of output ground trutth [82]"
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


def myresnet50(device, num_output=79, use_pretrained=True, num_views=1):

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    import resnet_multi_view
    #model = resnet_multi_view.resnet50(pretrained=True,num_classes=num_output)
    model = resnet_multi_view.resnet50(
        pretrained=use_pretrained, num_views=num_views)
    #model = torchvision.models.resnet50(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features * num_views

    # print(num_ftrs)    # 2048 : features
    # print(model.fc.out_features)   #1000 resnet deafult: number of output classes
    # change output number of classes
    model.fc = nn.Linear(num_ftrs, num_output)

    model = model.to(device)
    # model = nn.DataParallel(model)     # multi GPU
    return model


def train_model(device, model, dataloader, criterion, optimiser, scheduler, args,):
    import time
    import copy
    
    since = time.time()

    num_epochs = args.num_epochs
    vis = visdom.Visdom()
    win = vis.line(X=np.array([0]),Y=np.array([0]),opts=dict(
        title='dataset%d    batch:%d    output:%d'%(args.dataset_size,args.batch_size,args.num_output),
        legend=['train', 'val']))
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.

            for index, imgs, parameters in dataloader[phase]:
                # print(len(imgs))     # num_views
                # batch size 6
                # print(index)   # tensor([0, 1, 2, 3, 4, 5])

                # print(inputs_1.size())     # torch.Size([6, 3, 400, 400])  frontal view
                # print(inputs_2.size())     # torch.Size([6, 3, 400, 400])  side view
                # print(parameters.shape)
                #print('batch reading complete')

                inputs = torch.FloatTensor([])
                for k in range(0, len(imgs)):    # go through the views
                    inputs = torch.cat((inputs, imgs[k]), 0)
                    '''
                    print(imgs[k].shape)     # torch.Size([6 (batch), 3, 400, 400])
                    for count in range(0,args.batch_size):
                        #print(index[count])
                        torch.set_printoptions(profile='full')
                        print(imgs[k][count][0].shape)
                        input()
                        print(torch.all(torch.eq(imgs[k][count][0],imgs[k][count][1])))
                        print(torch.all(torch.eq(imgs[k][count][0],imgs[k][count][2])))
                    
                        input() 
                        torch.set_printoptions(profile='default')
                        check = torchvision.transforms.ToPILImage()(imgs[k][count])
                        #check.show()
                        
                    '''

                # inputs = torch.cat((inputs[0],inputs[1]),0)   # torch.Size([12, 3, 400, 400])

                inputs = inputs.to(device)

                parameters = parameters.float()  # from double to float
                parameters = parameters[:,:args.num_output]
                '''
                tmp = torch.FloatTensor([])
                for row in range(0, parameters.shape[0]):
                    tmp = torch.cat((tmp, parameters[row][:args.num_output].unsqueeze_(0)), 0)
                parameters = tmp
                '''
                parameters = parameters.to(device)
                # zero the parameter gradients
                optimiser.zero_grad()

                # forward
                # track history if only in train
                counter = 0
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs.shape)    # torch.Size([batch x num_views, 3, 400, 400])
                    # input()
                    outputs = model(inputs)

                    loss = criterion(outputs, parameters)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if (epoch+1) % 5 == 0:
                print(phase, '-----------------------')
                for k in range(2):
                    print('actual')
                    print(parameters[k][72:].to("cpu").numpy(),)
                    print('predict')
                    print(outputs[k][72:].to("cpu").detach().numpy(),)
                    print()
            if phase == 'train':
                scheduler.step()
            #epoch_loss = running_loss / data_size[phase]
            dataset_size = split_dataset(args.dataset_size)
            epoch_loss = running_loss / dataset_size[phase]

            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss))
            vis.line(
                X=np.array([epoch]),
                Y=np.array([epoch_loss]),
                name=phase,
                win=win,
                update='append'
            )


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

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
    #args.num_output = 20752
    print('number of output: ' , args.num_output)
    
    parent_dic = "/home/yifu/workspace/data_smpl/A_pose_5/male/noisy_original"
    print ('path: ', parent_dic)
    dataloader = load_data(args.dataset_size, parent_dic, args)

    model = myresnet50(device, num_output=args.num_output,
                       use_pretrained=True, num_views=args.num_views)
    criterion = nn.MSELoss()
    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)
    model = train_model(device, model, dataloader, criterion,
                        optimiser, exp_lr_scheduler, args)

    save_name = 'trained_resnet_%d_%d.pth'%(args.num_output,args.dataset_size)
    save_path = os.path.join(parent_dic, save_name)
    
    #save_path = "./trained_resnet.pth"

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
