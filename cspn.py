import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import time
import pickle
import matplotlib.pyplot as plt
from PIL import Image
cmap = plt.cm.jet

from nyu_dataloader import DataLoader
import model


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss


def create_depth_color(depth):
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth = (255 * cmap(depth_relative)[:, :, :3])
    return depth


def save_image(model, x, y, batch, mode="train"):
    pred = model(x)
    npimg = pred.cpu().detach().numpy()
    depth = create_depth_color(np.transpose(npimg[0], [1,2,0])[:, :, 0])
    target = create_depth_color(np.transpose(y[0].cpu().numpy(), [1,2,0])[:, :, 0])
    orig = 255 * np.transpose(x[0].cpu().numpy(), [1,2,0])

    img = np.concatenate((orig, target, depth), axis =1)
    
    img = Image.fromarray(img.astype('uint8'))
    img.save('saved_images/%s_image_%d.jpg'%(mode, batch))
    
def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train_loader = DataLoader("../cnn_depth_tensorflow/data/nyu_datasets/")
val_loader = DataLoader("../cnn_depth_tensorflow/data/nyu_datasets/", mode="val")

model = model.resnet50().double()
model = nn.DataParallel(model).cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
criterion_L1 = MaskedL1Loss()
criterion_MSE = MaskedMSELoss()

batch_size = 4
num_epochs = 40
num_batches = len(train_loader)//batch_size


for epoch in range(num_epochs):
    for iter_ in range(num_batches):
        adjust_learning_rate(optimizer, iter_, 0.001)
        x, y = next(train_loader.get_one_batch(batch_size))
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        save_image(model, x, y, iter_)
        torch.cuda.synchronize()

        optimizer.zero_grad()
        loss_L1 = criterion_L1(pred, y)
        loss_MSE = criterion_MSE(pred, y)
        print("Epoch {}, Batch {}/{}, L1 = {}, MSE = {}".format(epoch, iter_, num_batches, 
                                                                loss_L1.item(), loss_MSE.item()))
        
        loss_L1.backward()
        optimizer.step()
    
