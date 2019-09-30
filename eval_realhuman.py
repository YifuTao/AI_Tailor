import torchvision
import torch
from train_model import myresnet50, load_data, parse_args
from os.path import join
import os
from PIL import Image
from torchvision.transforms import ToTensor
import csv
import sys
sys.path.append("/home/yifu/workspace/smpl")
# from smpl_webuser.serialization import load_model
# from calculate_circumference import get_measurements
from body_measurements import vertex2measurements
from SMPL_Pytorch import par_to_mesh, decompose_par




def evaluate_model(model, num_views, path, device, args, normalise_scale=1,):
    import pickle
    was_training = model.training
    model.eval()

    with torch.no_grad():
        
        m = pickle.load(open('models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % args.gender[0]))
        csv_name = os.path.join(path, 'real_male.csv')
        real_height = [1.760837, 1.696549, 1.779879, 1.766942, 1.780513]
        # Check real human
        with open(csv_name, mode='w') as csv_file:
            for i in range(0, args.dataset_size):
                inputs = torch.FloatTensor([])
                for k in range(0, num_views):    # go through the views
                    img_name = os.path.join(path, '%d_%d.png' % (i, k))  # image
                    img = Image.open(img_name)
                    img = ToTensor()(img)[0]
                    '''
                    for x in range(img.shape[0]):
                        for y in range(img.shape[1]):
                            if img[x][y] > 0.02:
                                img[x][y] = 1
                    '''
                    img.unsqueeze_(0)

                    img = torch.cat((img, img, img), 0)
                    img.unsqueeze_(0)
                    inputs = torch.cat((inputs, img), 0)

                inputs = inputs.to(device)
                par_prd = model(inputs)
                batch = par_prd.shape[0]
                rots, poses, betas = decompose_par(par_prd)
                mesh_prd = par_to_mesh(args.gender, rots, poses, betas)

                par_gt = par_gt.float()  # from double to float
                par_gt = par_gt.to(device)
                X, Y, Z = [mesh_prd[:,:, 0], mesh_prd[:,:, 1], mesh_prd[:,:, 2]]
                h_prd, w_prd, c_prd, n_prd, a_prd = vertex2measurements(X, Y, Z)

                vertices = torch.reshape(mesh_prd, (batch, -1))

                ratio = real_height[i] / h_prd

                csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
                csv_writer.writerow([i,'Height',float(real_height[i])])
                csv_writer.writerow([i,'Waist',float(w_prd*ratio),])
                csv_writer.writerow([i,'Chest',float(c_prd*ratio),])
                csv_writer.writerow([i,'Neck',float(n_prd*ratio)])
                csv_writer.writerow([i,'Arm',float(a_prd*ratio),])
                csv_writer.writerow(['\n'])
                # mesh generator
                outmesh_path = join(path, '%d.obj'%i)
                with open(outmesh_path, 'w') as fp:
                    for v in mesh_prd[0]:
                        #fp.write( 'v %f %f %f\n' % ( float(v[0]),float(v[1]), float(v[2])) )
                        fp.write( 'v %f %f %f\n' % (v[0], v[1],v[2]))
                    for f in m['f']+1: # Faces are 1-based, not 0-based in obj files
                        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
        csv_file.close()

    model.train(mode=was_training)


def main():
    args = parse_args()
    args. dataset_size = 5

    device = torch.device("cuda:%d" % 
                          args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    num_output = 82
    model = myresnet50(device, num_output=num_output,
                       use_pretrained=True, num_views=args.num_views)
    
    # folder: network weights
    parent_dic = "/home/yifu/workspace/data/test/model_1"
    save_name = 'data:%d.pth' % (100000)
    path = os.path.join(parent_dic, save_name)
    model.load_state_dict(torch.load(path))
    # folder: image 
    path = "/home/yifu/workspace/data/test/model_1"
    evaluate_model(model, args.num_views, path, device, args)


if __name__ == "__main__":
    main()
