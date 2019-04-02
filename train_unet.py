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
from tensorboardX import SummaryWriter
cmap = plt.cm.jet

from nyu_dataloader import DataLoader
import model
from unet import UNet

writer = SummaryWriter("runs/run1")

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


class ScaleInvariantError(nn.Module):
    def __init__(self, lamada=0.5):
        super(ScaleInvariantError, self).__init__()
        self.lamada = lamada
        return

    def forward(self, y_true, y_pred):
        first_log = torch.log(torch.clamp(y_pred, 0.0001))
        second_log = torch.log(torch.clamp(y_true, 0.0001))
        d = first_log - second_log
        loss = torch.mean(d * d) - self.lamada * torch.mean(d) * torch.mean(d)
        return loss

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
    """Sets the learning rate to the initial LR decayed by 2 every 5 epochs"""
    lr = lr_init * (0.5 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, val_loader, model, criterion_L1, criterion_MSE, 
          criterion_berHu, optimizer, epoch, batch_size):
    
    model.train()  # switch to train mode
    eval_mode = False
    init_lr = optimizer.param_groups[0]['lr']
    
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
        loss_SI = criterion_SI(pred, target)

        writer.add_scalar('L1 Loss', loss_L1.item(), 25*epoch + iter_ + 1)
        writer.add_scalar('MSE Loss', loss_MSE.item(), 25*epoch + iter_ + 1)
        writer.add_scalar('SI Loss', loss_SI.item(), 25*epoch + iter_ + 1)
        writer.add_scalar('berHu Loss', loss_berHu.item(), 25*epoch + iter_ + 1)

        writer.add_scalars('loss/metrics', { 
                              "L1": loss_L1.item(), "MSE": loss_MSE.item(), 
                              "SI": loss_SI.item(), "berHu": loss_berHu.item()}
                              , 25 * epoch + iter_)
        loss_gen = loss_SI + 5 * loss_L1
        loss_gen.backward()
        optimizer.step()
        
        if (iter_ + 1) % 10 == 0:
            save_image(model, input, target, iter_)
            print('Train Epoch: {} Batch: [{}/{}], SI: {:0.4f}, L1 ={:0.3f}, MSE={:0.3f}, berHu={:0.3f}'.format(
                epoch, iter_ + 1, num_batches, loss_SI.item(),
                loss_L1.item(), loss_MSE.item(), loss_berHu.item()))


train_loader = DataLoader("../cnn_depth_tensorflow/data/nyu_datasets/")
val_loader = DataLoader("../cnn_depth_tensorflow/data/nyu_datasets/", mode="val")

model = UNet(3, 1).double()
model = nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002, weight_decay=1e-4)

criterion_L1 = MaskedL1Loss()
criterion_berHu = berHuLoss()
criterion_MSE = MaskedMSELoss()
criterion_SI = ScaleInvariantError()

batch_size = 8
num_epochs = 40
num_batches = len(train_loader)//batch_size

for epoch in range(25):
    adjust_learning_rate(optimizer, epoch, 0.001)
    train(train_loader, val_loader, model, criterion_L1, criterion_MSE, criterion_berHu, 
          optimizer, epoch, batch_size)
     
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './unet_si.pth')


