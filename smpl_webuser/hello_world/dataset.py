import os
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import ToTensor,ToPILImage
from skimage import io, transform
import pickle

class HumanTestSet(data.Dataset):
    def __init__(self, root,data_size, transform=None, num_views=1,normalise_scale=1):
        #super(HumanTestSet, self).__init__()
        #super(HumanTestSet, self).__init__(root, data_size, transform)
        self.root = root
        self.transform = transform
        self.data_size = data_size    # train/val
        self.normalise_scale = normalise_scale
        self.num_views = num_views

    def __len__(self):
        return self.data_size 

    def __getitem__(self, index):
        #parameters
        par_name = os.path.join(self.root, '%d' % index) 
        infile = open(par_name, 'rb')
        par = pickle.load(infile)
        #par = pickle.load(infile, encoding='latin1') #pickle in py2 but using py3 now
        infile.close()

        # image
        imgs=[]
        for i in range(0,self.num_views):
            img_name = os.path.join(self.root, '%d_%d.png' % (index,i))    #image
            img = Image.open(img_name)
            '''
            img = ToTensor()(img)
            img[0] = torch.ones_like(img[0])
            height = par[79]
            img[0] *= height
            img = ToPILImage()(img)
            '''
            imgs.append(self.transform(img))
        

   
        

        return index,imgs,par     #return imgs and label



    