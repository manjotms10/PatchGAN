import numpy as np
import os
import glob
# import cv2
import torch
from PIL import Image
from skimage.util import img_as_float
from skimage.transform import resize

class DataLoader():
    '''
    raw_data_dir = dir containing extracted raw KITTI data (folder containing 2011-09-26 etc.)
    depth_maps_dir = dir containing extracted depth maps
    '''
    
    def __init__(self, raw_images_path, depth_images_path, mode="train"):
        
        self.mode = mode
        
        if self.mode == "train":
            with open('eigen_train_files.txt', 'r') as f:
                self.files = f.readlines()
        elif self.mode == "test":
            with open('eigen_test_files.txt', 'r') as f:
                self.files = f.readlines()
        elif self.mode == "val":
            with open('eigen_val_files.txt', 'r') as f:
                self.files = f.readlines()
            
        self.data = []
        for l in self.files:
            s = l.split()
            for im in s:
                self.data.append(raw_images_path + im)

        self.imgs, self.labels = [], []
        
        for img_name in self.data:
            img_name = img_name.split('.')[0] + '.png'
            tokens = img_name.split('/')
            path = depth_images_path + 'train/' + tokens[-4] + '/proj_depth/groundtruth/' + tokens[-3] + '/' + tokens[-1]
            path = path.split('.')[0] + '.png'
            if os.path.exists(path) and os.path.exists(img_name):
                self.imgs.append(img_name)
                self.labels.append(path)
                
        print('Found %d Images %d'%(len(self.imgs), len(self.labels)))
    
    def __len__(self):
        return len(self.imgs)
    
    def load_data(self, img_file, label_file):
        x = Image.open(img_file) 
        y = Image.open(label_file)
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)
        x = resize(x, (90, 270), anti_aliasing=True)
        y = resize(y, (90, 270), anti_aliasing=True)
        x = x / 255.0
        y[y<1] = y
        y = np.log(y)
        y = y / np.max(y)
        return x, y
        
    def get_one_batch(self, batch_size = 64):
        images = []
        labels = []

        while True:
            idx = np.random.choice(len(self.imgs), batch_size)
            for i in idx:
                x, y = self.load_data(self.imgs[i], self.labels[i])
                images.append(x)
                labels.append(y)
            yield torch.from_numpy(np.array(images)).permute(0,3,1,2), torch.from_numpy(np.array(labels)).permute(0,3,1,2)
