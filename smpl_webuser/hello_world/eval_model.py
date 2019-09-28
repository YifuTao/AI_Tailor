import torch
from train_model import myresnet50, load_data, parse_args
import os
from calculate_circumference import get_measurements
from smpl_webuser.serialization import load_model
import numpy as np


def evaluate_model(m, model, dataloader, num_views, path, device, args, normalise_scale=1, num_images=10):
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
        for index, imgs, parameters in dataloader['val']:
            count = count + 1
            inputs = torch.FloatTensor([])
            for k in range(0, len(imgs)):    # go through the views
                inputs = torch.cat((inputs, imgs[k]), 0)

            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            outputs = np.squeeze(outputs, axis=0)
            outputs = outputs[:args.num_output]
            parameters = parameters.float().cpu().numpy()  # from double to float
            parameters = np.squeeze(parameters, axis=0)
            parameters = parameters[:args.num_output]
            h_pre, w_pre, c_pre, n_pre, a_pre = get_measurements(m, outputs)
            h_gt, w_gt, c_gt, n_gt, a_gt = get_measurements(m, parameters)

            h_error = h_error + abs(h_pre - h_gt)
            w_error = w_error + abs(w_pre - w_gt)
            c_error = c_error + abs(c_pre - c_gt)
            n_error = n_error + abs(n_pre - n_gt)
            a_error = a_error + abs(a_pre - a_gt)
            '''
            inputs = inputs.to(device)
            parameters = parameters.to(device)
            parameters = [i * normalise_scale for i in parameters]
            outputs = model(inputs)
            batch_size = inputs.size()[0]
            

            for j in range(batch_size):
                print('predict')
                print(outputs[j][72:])
                print('actual')
                print(parameters[j][72:])
            '''
    print(h_error,w_error,c_error,n_error,a_error,)
    print('number of images:',count)
    print('error in milimeters')
    print(h_error/count*1000,w_error/count*1000,c_error/count*1000,n_error/count*1000,a_error/count*1000,)
    model.train(mode=was_training)


def main():
    args = parse_args()
    args.dataset_size = 100000
    args.batch_size = 1
    args.num_output = 82
    gender = 'male'    #   female
    m = load_model('../../models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % gender[0])
    parent_dic = "/home/yifu/workspace/data_smpl/A_pose_5/male/noise_free"
    dataloader = load_data(args.dataset_size, parent_dic, args)
    device = torch.device("cuda:%d" %
                          args.gpu if torch.cuda.is_available() else "cpu")
    model = myresnet50(device, num_output=args.num_output,
                       use_pretrained=True, num_views=args.num_views)
    #model = myresnet50(num_output=80)
    save_name = 'trained_resnet_%d_%d.pth' % (args.num_output, args.dataset_size)
    path = os.path.join(parent_dic, save_name)
    model.load_state_dict(torch.load(path))

    path = parent_dic
    evaluate_model(m, model, dataloader, args.num_views, path, device, args)


if __name__ == "__main__":
    main()
