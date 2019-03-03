import numpy as np
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
from unet import UNet

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential (
                    nn.Conv2d(1, 32, 4, 1, 0),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 32, 3, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 32, 3, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True)
                )
        self.out = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 32)
        x = self.out(x)
        return torch.sigmoid(x)


def set_requires_grad(net, requires_grad=False):
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad


def train(train_loader, val_loader, model, discriminator, criterion_L1, criterion_MSE, 
          criterion_berHu, criterion_GAN, optimizer, optimizer_D, epoch, batch_size):
    
    model.train()  # switch to train mode
    eval_mode = False
    num_batches = len(train_loader)// batch_size
    init_lr = optimizer.param_groups[0]['lr']
    
    valid_T = torch.ones(batch_size, 1).cuda().double()
    zeros_T = torch.zeros(batch_size, 1).cuda().double()
    
    for iter_ in range(num_batches):
        input, target = next(train_loader.get_one_batch(batch_size))
        input, target = input, target
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        
        optimizer.zero_grad()
        
        pred = model(input)
        
        loss_L1 = criterion_L1(pred, target)
        loss_MSE = criterion_MSE(pred, target)
        loss_berHu = criterion_berHu(pred, target)
        
        set_requires_grad(discriminator, False)
        
        loss_adv = 0
        
        for a in range(12):
            for b in range(16):
                row = 19 * a
                col = 19 * b
                patch_fake = pred[:, :, row:row+19, col:col+19]
                pred_fake = discriminator(patch_fake)
                loss_adv += criterion_GAN(pred_fake, valid_T)
    
        loss_gen = loss_L1 + loss_adv
        loss_gen.backward()
        optimizer.step()
        
        set_requires_grad(discriminator, True)
        optimizer_D.zero_grad()
        loss_D = 0
        for a in range(12):
            for b in range(16):
                row = 19 * a
                col = 19 * b
                patch_fake = pred[:, :, row:row+19, col:col+19]
                patch_real = target[:, :, row:row+19, col:col+19]
                pred_fake = discriminator(patch_fake.detach())
                pred_real = discriminator(patch_real)
                loss_D_fake = criterion_GAN(pred_fake, zeros_T)
                loss_D_real = criterion_GAN(pred_real, valid_T)
                loss_D += 0.5 * (loss_D_fake + loss_D_real)
        
        loss_D.backward()
        optimizer_D.step()

        torch.cuda.synchronize()
        if (iter_ + 1) % 10 == 0:
            save_image(model, input, target, iter_)
            print('Train Epoch: {} Batch: [{}/{}], ADV:{:0.3f} L1 ={:0.3f}, MSE={:0.3f}, berHu={:0.3f}, Disc:{:0.3f}'.format(
                epoch, iter_ + 1, num_batches, loss_adv.item(),
                loss_L1.item(), loss_MSE.item(), loss_berHu.item(), loss_D.item()))


train_loader = DataLoader("../cnn_depth_tensorflow/data/nyu_datasets/")
val_loader = DataLoader("../cnn_depth_tensorflow/data/nyu_datasets/", mode="val")

model = UNet(3, 1).double()
model = nn.DataParallel(model).cuda()

discriminator = Discriminator().double()
discriminator.apply(weights_init)
discriminator = discriminator.cuda()

optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=0.9, weight_decay=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=4 * 0.001)

criterion_L1 = MaskedL1Loss()
criterion_berHu = berHuLoss()
criterion_MSE = MaskedMSELoss()
criterion_GAN = nn.BCELoss()
criterion = nn.L1Loss()

batch_size = 8
num_epochs = 40
num_batches = len(train_loader)//batch_size

for epoch in range(10):
    train(train_loader, val_loader, model, discriminator, criterion_L1, criterion_MSE, criterion_berHu, 
          criterion_GAN, optimizer, optimizer_D, epoch, batch_size)
     
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './unet.pth')


