import os
import pickle

import numpy as np
import glob
import torch
from PIL import Image
from skimage.util import img_as_float
from skimage.transform import resize
from torchvision import transforms


class DataLoader():
    '''
    raw_data_dir = dir containing extracted raw NYU data (folder containing 2011-09-26 etc.)
    '''
    
    def __init__(self, raw_images_path, mode="train"):
        
        self.mode = mode
        self.raw_images_path = raw_images_path
        self.split_file = "nyu_split.pkl"
        
        if not os.path.isfile(self.split_file):
            self.train_test_split()
        
        with open(self.split_file, "rb") as split_f:
            split = pickle.load(split_f)
        
        if self.mode == "train":
            self.images = split["train"]
        else:
            self.images = split["val"]
        
        print('Found Images %d'%(len(self.images)))

    def load_data(self, img_file, label_file):
        x = Image.open(img_file) 
        y = Image.open(label_file)
        x = np.array(x)
        y = np.array(y)

        x = resize(x, (228, 304), anti_aliasing=True)
        y = resize(y, (228, 304), anti_aliasing=True)
        x = img_as_float(x)
        y = img_as_float(y)
        
        return x, y
    
    def __len__(self):
        return len(self.images)
    
    def train_test_split(self):
        images = [x.split('/')[-1].split('.')[0] for x in sorted(glob.glob(self.raw_images_path + '/*.jpg'))]

        total = len(images)
        indexes = np.random.permutation(total)

        split = int(0.9 * total)
        train = images[:split]
        val = images[split:]
        out = {'train': train, 'val': val}
        with open("nyu_split.pkl", "wb") as file:
            pickle.dump(out, file)
        
    def get_one_batch(self, batch_size = 16):
        images = []
        labels = []

        while True:
            idx = np.random.choice(len(self.images), batch_size)
            for i in idx:
                image = os.path.join(self.raw_images_path, self.images[i] + ".jpg")
                label = os.path.join(self.raw_images_path, self.images[i] + ".png")
                x, y = self.load_data(image, label)
                #x, y = self.train_transform(x, y)
                y= np.expand_dims(y, axis=0)
                images.append(x)
                labels.append(y)
            yield torch.from_numpy(np.array(images)).permute(0, 3, 1, 2), torch.from_numpy(np.array(labels))

            
    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            #transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np