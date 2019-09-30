import torch
from train_model import myresnet50, load_data, parse_args
import os
from body_measurements import get_measurements
import numpy as np

import sys
sys.path.append('functions')
from SMPL_Pytorch import par_to_mesh, decompose_par
from body_measurements import vertex2measurements


def evaluate_model(model, dataloader, num_views, path, device, args, normalise_scale=1,):
    import pickle
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # for i, (inputs, parameters) in enumerate(dataloader['val']):
        count = 0
        h_error = 0
        w_error = 0
        c_error = 0
        n_error = 0
        a_error = 0
        for index, imgs, par_gt in dataloader['val']:
            count = count + 1
            inputs = torch.FloatTensor([])
            for k in range(0, len(imgs)):    # go through the views
                inputs = torch.cat((inputs, imgs[k]), 0)
            
            inputs = inputs.to(device)
            par_prd = model(inputs)
            batch = par_prd.shape[0]

            rots, poses, betas = decompose_par(par_prd)
            mesh_prd = par_to_mesh(args.gender, rots, poses, betas)
            X, Y, Z = [mesh_prd[:,:, 0], mesh_prd[:,:, 1], mesh_prd[:,:, 2]]
            h_prd, w_prd, c_prd, n_prd, a_prd = vertex2measurements(X, Y, Z)

            par_gt = par_gt.float()  # from double to float
            par_gt = par_gt.to(device)
            rots, poses, betas = decompose_par(par_gt)
            mesh_gt = par_to_mesh(args.gender, rots, poses, betas)
            X, Y, Z = [mesh_gt[:,:, 0], mesh_gt[:,:, 1], mesh_gt[:,:, 2]]
            h_gt, w_gt, c_gt, n_gt, a_gt = vertex2measurements(X, Y, Z)

            '''
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            outputs = np.squeeze(outputs, axis=0)
            outputs = outputs[:args.num_output]
            parameters = parameters.float().cpu().numpy()  # from double to float
            parameters = np.squeeze(parameters, axis=0)
            parameters = parameters[:args.num_output]
            h_pre, w_pre, c_pre, n_pre, a_pre = get_measurements(m, outputs)
            h_gt, w_gt, c_gt, n_gt, a_gt = get_measurements(m, parameters)
            '''
            h_error = h_error + float(abs(h_prd - h_gt))
            w_error = w_error + float(abs(w_prd - w_gt))
            c_error = c_error + float(abs(c_prd - c_gt))
            n_error = n_error + float(abs(n_prd - n_gt))
            a_error = a_error + float(abs(a_prd - a_gt))
            
    print(h_error,w_error,c_error,n_error,a_error,)
    print 'number of images:',count
    print('error in milimeters')
    print 'height   waist   chest   neck    arm'
    print '%.3f    %.3f   %.3f   %.3f   %.3f'%(h_error/count*1000,w_error/count*1000,
        c_error/count*1000,n_error/count*1000,a_error/count*1000)
    model.train(mode=was_training)


def main():
    args = parse_args()
    #args.dataset_size = 100000
    print('-----------------------------------------------------------')
    print('Dataset size: ', args.dataset_size)
    args.batch_size = 1
    gender = 'male'    #   female
    print('Gender: ', gender)
    device = torch.device("cuda:%d" %
                          args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    parent_dic = "/home/yifu/workspace/data/synthetic/noise_free"
    print('Data path: ', parent_dic)
    dataloader = load_data(args.dataset_size, parent_dic, args)

    
    model = myresnet50(device, num_output=args.num_output,
                       use_pretrained=True, num_views=args.num_views)
    
    # save_name = 'out:%d_data:%d_par_w:%.1f.pth'%(args.num_output,args.dataset_size, args.par_loss_weight)

    # folder: network weights
    parent_dic = "/home/yifu/workspace/data/test/model_1"
    save_name = 'data:%d.pth' % (100000)
    save_path = os.path.join(parent_dic, save_name)
    print('Load state dict from save path: ', save_path)
    model.load_state_dict(torch.load(save_path))
    print('-----------------------------------------------------------')

    if raw_input('Confirm the above setting? (yes/no): ')!='yes':
        print('Terminated')
        exit()
    print('validation starts')
    print('------------------------')
    path = parent_dic
    evaluate_model(model, dataloader, args.num_views, path, device, args)


if __name__ == "__main__":
    main()
