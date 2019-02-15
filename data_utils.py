import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

class KittiDataLoader():
    '''
    raw_data_dir = dir containing extracted raw KITTI data (folder containing 2011-09-26 etc.)
    depth_maps_dir = dir containing extracted depth maps
    '''
    
    def __init__(self, raw_data_dir, depth_maps_dir):
        
        depth_search = "2011_*_*_drive_*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
        self.depth_files = sorted(glob.glob(os.path.join(depth_maps_dir, 'train', depth_search)))
        
        self.img_search = "2011_*_*/2011_*_*_drive_*_sync/image_0[2,3]/data/*.png"
        self.img_files = sorted(glob.glob(os.path.join(raw_data_dir, self.img_search)))
        
        self.labels=[]
        self.imgs = []
        
        print('Loading Data ...')
        for img_name in self.img_files:
            tokens = img_name.split('/')
            path = depth_maps_dir + 'train/' + tokens[-4] + '/proj_depth/groundtruth/' + tokens[-3] + '/' + tokens[-1]
            if os.path.exists(path):
                self.imgs.append(img_name)
                self.labels.append(path)
        
        print('Found {} Images and {} Depth Maps'.format(len(self.imgs), len(self.labels)))
        
    def get_one_batch(self, batch_size = 64):
        train_images = []
        train_labels = []

        while True:
            idx = np.random.choice(len(self.labels), batch_size)
            for i in idx:
                train_images.append(cv2.imread(self.img_files[i]).astype(np.float32)/255)
                train_labels.append(cv2.imread(self.labels[i]).astype(np.float32))
            yield np.array(train_images), np.array(train_labels)
