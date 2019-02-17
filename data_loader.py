import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset

import transforms


class DataLoader():
    '''
    raw_data_dir = dir containing extracted raw KITTI data (folder containing 2011-09-26 etc.)
    depth_maps_dir = dir containing extracted depth maps
    '''
    
    def __init__(self, raw_images_path, depth_images_path):
        
        with open('eigen_train_files.txt', 'r') as f:
            self.train_files = f.readlines()
        with open('eigen_test_files.txt', 'r') as f:
            self.test_files = f.readlines()
        with open('eigen_val_files.txt', 'r') as f:
            self.val_files = f.readlines()
            
        self.train_data = []
        self.test_data = []
        self.val_data = []

        for l in self.train_files:
            self.train_data.append(raw_images_path + l.split(' ')[0])
            self.train_data.append(raw_images_path + l.split(' ')[1])

        for l in self.test_files:
            self.test_data.append(raw_images_path + l.split(' ')[0])
            self.test_data.append(raw_images_path + l.split(' ')[1])

        for l in self.val_files:
            self.val_data.append(raw_images_path + l.split(' ')[0])
            self.val_data.append(raw_images_path + l.split(' ')[1])
        
        
        self.train_imgs, self.train_labels = [], []
        for img_name in self.train_data:
            img_name = img_name.split('.')[0] + '.png'
            tokens = img_name.split('/')
            path = depth_images_path + 'train/' + tokens[-4] + '/proj_depth/groundtruth/' + tokens[-3] + '/' + tokens[-1]
            path = path.split('.')[0] + '.png'
            if os.path.exists(path) and os.path.exists(img_name):
                self.train_imgs.append(img_name)
                self.train_labels.append(path)

        self.val_imgs, self.val_labels = [], []
        for img_name in self.val_data:
            img_name = img_name.split('.')[0] + '.png'
            tokens = img_name.split('/')
            path = depth_images_path + 'train/' + tokens[-4] + '/proj_depth/groundtruth/' + tokens[-3] + '/' + tokens[-1]
            path = path.split('.')[0] + '.png'
            if os.path.exists(path) and os.path.exists(img_name):
                self.val_imgs.append(img_name)
                self.val_labels.append(path)

        self.test_imgs, self.test_labels = [], []
        for img_name in self.test_data:
            img_name = img_name.split('.')[0] + '.png'
            tokens = img_name.split('/')
            path = depth_images_path + 'train/' + tokens[-4] + '/proj_depth/groundtruth/' + tokens[-3] + '/' + tokens[-1]
            path = path.split('.')[0] + '.png'
            if os.path.exists(path) and os.path.exists(img_name):
                self.test_imgs.append(img_name)
                self.test_labels.append(path)
                
        print('Found %d Training Images %d'%(len(self.train_imgs), len(self.train_labels)))
        print('Found %d Validation Images %d'%(len(self.val_imgs), len(self.val_labels)))
        print('Found %d Test Images %d'%(len(self.test_imgs), len(self.test_labels)))

    def get_one_batch(self, batch_size = 64, split='train'):
        train_images = []
        train_labels = []

        while True:
            idx = np.random.choice(len(self.train_imgs), batch_size)
            for i in idx:
                x, y = cv2.imread(self.train_imgs[i]).astype(np.float32), cv2.imread(self.train_labels[i]).astype(np.float32)
                x, y = cv2.resize(x, (90, 270)), cv2.resize(y, (90, 270))
                train_images.append(x)
                train_labels.append(y)

            images = torch.from_numpy(np.array(train_images)).permute(0, 3, 2, 1)
            labels = torch.from_numpy(np.array(train_labels)).permute(0, 3, 2, 1)
            yield images, labels


class KittiData(Dataset):

    def __init__(self, root_dir, depth_map_dir, mode='train', size=(385, 513)):
        super(KittiData, self).__init__()
        self.root_dir = root_dir
        self.depth_map_dir = depth_map_dir
        self.mode = mode
        self.im_gt_paths = None
        self.size = size

        if self.mode == 'train':
            with open('eigen_train_files.txt', 'r') as f:
                self.files = f.readlines()
        elif self.mode == 'test':
            with open('eigen_test_files.txt', 'r') as f:
                self.files = f.readlines()
        elif self.mode == 'val':
            with open('eigen_val_files.txt', 'r') as f:
                self.files = f.readlines()
        else:
            print('no mode named as ', mode)
            exit(-1)

        for l in self.files:
            file_names = l.split()
            self.im_gt_paths.append(file_names[0])
            self.im_gt_paths.append(file_names[1])

    def __len__(self):
        return len(self.im_gt_paths)

    def train_transform(self, im, gt):
        im = np.array(im).astype(np.float32)
        gt = np.array(gt).astype(np.float32)

        s = np.random.uniform(1.0, 1.5)  # random scaling
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.Resize(460 / 240, interpolation='bilinear'),
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.size),
            transforms.HorizontalFlip(do_flip)
        ])

        im_ = transform(im)
        im_ = color_jitter(im_)

        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 100.0 * s
        im_ = torch.Tensor(im_)
        gt_ = torch.Tensor(gt_)

        gt_ = gt_.unsqueeze(0)

        return im_, gt_

    def val_transform(self, im, gt):
        im = np.array(im).astype(np.float32)
        gt = np.array(gt).astype(np.float32)

        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.Resize(460 / 240, interpolation='bilinear'),
            transforms.CenterCrop(self.size)
        ])

        im_ = transform(im)
        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 100.0
        im_ = torch.Tensor(im_)
        gt_ = torch.Tensor(gt_)

        gt_ = gt_.unsqueeze(0)
        return im_, gt_

    def load(self, path, rgb=True):

        with open(path, 'rb') as f:
            img = Image.open(f)
            if rgb:
                return img.convert('RGB')
            else:
                return img.convert('I')

    def get_paths(self, img_name):
        img_name = img_name.split('.')[0] + '.png'
        tokens = img_name.split('/')
        depth_path = self.depth_map_dir + 'train/' + tokens[-4] + '/proj_depth/groundtruth/' + tokens[-3] + '/' + tokens[-1]
        depth_path = depth_path.split('.')[0] + '.png'
        img_path = self.root_dir + img_name
        return img_path, depth_path

    def __getitem__(self, idx):
        im_path, gt_path = self.get_paths(self.im_gt_paths[idx])

        im = self.load(im_path)
        gt = self.load(gt_path, rgb=False)

        if self.mode == 'train':
            im, gt = self.train_transform(im, gt)
        else:
            im, gt = self.val_transform(im, gt)

        return im, gt
