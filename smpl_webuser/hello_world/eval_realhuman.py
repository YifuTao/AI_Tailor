import torchvision
import torch
from train_model import myresnet50, load_data, parse_args
import sys
sys.path.append("/home/yifu/workspace/smpl")
from smpl_webuser.serialization import load_model
from calculate_circumference import get_measurements
import os
from PIL import Image
from torchvision.transforms import ToTensor
import csv


def evaluate_model(m, model, num_views, path, device, args, normalise_scale=1,):
    import pickle
    was_training = model.training
    model.eval()

    with torch.no_grad():
        '''
        images = [f for f in listdir(path) if isfile(join(path, f))]
        for name in images:
            if name[-4:]!='.png':
                print('not png!')
                continue
        '''
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
                outputs = model(inputs)
                outputs.squeeze_(0)
                parameters = outputs.to('cpu').numpy()

                h, w, c, n, a = get_measurements(m, parameters)
                ratio = real_height[i] / h

                csv_writer = csv.writer(csv_file, delimiter=',',lineterminator='\n')
                csv_writer.writerow([i,'Height',real_height[i]])
                csv_writer.writerow([i,'Waist',w*ratio,])
                csv_writer.writerow([i,'Chest',c*ratio,])
                csv_writer.writerow([i,'Neck',n*ratio])
                csv_writer.writerow([i,'Arm',a*ratio,])
                csv_writer.writerow(['\n'])
                '''
                print(i)
                print(outputs.shape)
                print(outputs[-14:])
                real_height = int(input())
                print(real_height)
                tests = outputs.to("cpu").numpy()
                parameters = tests[:82]
                '''
                #pickle.dump(outputs.to("cpu").numpy(), open(
                    #'%s/%d' % (path, i), 'wb'), protocol=2)
        csv_file.close()

    model.train(mode=was_training)


def main():
    args = parse_args()

    #dataset_size = 1000
    #dataloader = load_data(args.dataset_size, args)
    device = torch.device("cuda:%d" % 
                          args.gpu if torch.cuda.is_available() else "cpu")
    num_output = 82
    model = myresnet50(device, num_output=num_output,
                       use_pretrained=True, num_views=args.num_views)
    gender = 'male'   
    m = load_model('../../models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % gender[0])
    #parent_dic = "/home/yifu/workspace/data_smpl/A_pose_3/new_vertexes"
    #parent_dic = "/home/yifu/workspace/data_smpl/A_pose_4/scaled"
    parent_dic = "/home/yifu/workspace/Data/MPI-FAUST/training/registrations_obj/male/test_model_2"
    #path = "./trained_resnet.pth"
    save_name = 'trained_resnet_%d_%d.pth' % (num_output,100000)
    path = os.path.join(parent_dic, save_name)
    model.load_state_dict(torch.load(path))

    #parent_dic = "/home/yifu/workspace/data_smpl/A_pose_3"
    #path = os.path.join(parent_dic, 'test')

    '''
    parent_dic = "/home/yifu/workspace/data_smpl/real_human"
    path = os.path.join(parent_dic, 'scaled')
    '''
    path = parent_dic
    
    evaluate_model(m, model, args.num_views, path, device, args)


if __name__ == "__main__":
    main()
