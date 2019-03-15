import numpy as np
import cv2
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
        
        self.train_images = sorted(glob.glob(raw_images_path + '/*.jpg'))
        self.train_labels = sorted(glob.glob(raw_images_path + '/*.png'))
        
        print('Found %d Images %d'%(len(self.train_images), len(self.train_labels)))
    
    def load_data(self, img_file, label_file):
        x = Image.open(img_file) 
        y = Image.open(label_file).convert('RGB')
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)

        x = resize(x, (228, 304), anti_aliasing=True)
        y = resize(y, (228, 304), anti_aliasing=True)
        x = img_as_float(x)/127.5 - 1
        y = img_as_float(y)/127.5 - 1
        
        return x, y
        
    def get_one_batch(self, batch_size = 16):
        images = []
        labels = []

        while True:
            idx = np.random.choice(len(self.train_images), batch_size)
            for i in idx:
                x, y = self.load_data(self.train_images[i], self.train_labels[i])
                #x, y = self.train_transform(x, y)
                images.append(x)
                labels.append(y)
            yield torch.from_numpy(np.array(images)).permute(0, 3, 1, 2), torch.from_numpy(np.array(labels)).permute(0, 3, 1, 2)

            
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

