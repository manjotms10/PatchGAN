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

from nyu_loader import DataLoader

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


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

    def forward(self, x):
        weights = torch.zeros(self.num_channels, 1, self.stride, self.stride)
        if torch.cuda.is_available():
            weights = weights.cuda()
        weights[:, :, 0, 0] = 1
        return F.conv_transpose2d(x, weights, stride=self.stride, groups=self.num_channels)


class Decoder(nn.Module):
    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                (module_name, nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size,
                                                 stride, padding, output_padding, bias=False)),
                ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
                ('relu', nn.ReLU(inplace=True)),
            ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))


class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
            ('unpool', Unpool(in_channels)),
            ('conv', nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
            ('relu', nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels // 2)
        self.layer3 = self.upconv_module(in_channels // 4)
        self.layer4 = self.upconv_module(in_channels // 8)


class FasterUpConv(Decoder):
    # Faster Upconv using pixelshuffle

    class faster_upconv_module(nn.Module):

        def __init__(self, in_channel):
            super(FasterUpConv.faster_upconv_module, self).__init__()

            self.conv1_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=3)),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv2_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(2, 3))),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv3_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(3, 2))),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv4_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=2)),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.ps = nn.PixelShuffle(2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            # print('Upmodule x size = ', x.size())
            x1 = self.conv1_(nn.functional.pad(x, (1, 1, 1, 1)))
            x2 = self.conv2_(nn.functional.pad(x, (1, 1, 0, 1)))
            x3 = self.conv3_(nn.functional.pad(x, (0, 1, 1, 1)))
            x4 = self.conv4_(nn.functional.pad(x, (0, 1, 0, 1)))

            x = torch.cat((x1, x2, x3, x4), dim=1)

            output = self.ps(x)
            output = self.relu(output)

            return output

    def __init__(self, in_channel):
        super(FasterUpConv, self).__init__()

        self.layer1 = self.faster_upconv_module(in_channel)
        self.layer2 = self.faster_upconv_module(in_channel // 2)
        self.layer3 = self.faster_upconv_module(in_channel // 4)
        self.layer4 = self.faster_upconv_module(in_channel // 8)


class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels // 2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
                ('batchnorm1', nn.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU()),
                ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels // 2)
        self.layer3 = self.UpProjModule(in_channels // 4)
        self.layer4 = self.UpProjModule(in_channels // 8)


class FasterUpProj(Decoder):
    # Faster UpProj decorder using pixelshuffle

    class faster_upconv(nn.Module):

        def __init__(self, in_channel):
            super(FasterUpProj.faster_upconv, self).__init__()

            self.conv1_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=3)),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv2_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(2, 3))),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv3_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(3, 2))),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv4_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=2)),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.ps = nn.PixelShuffle(2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            # print('Upmodule x size = ', x.size())
            x1 = self.conv1_(nn.functional.pad(x, (1, 1, 1, 1)))
            x2 = self.conv2_(nn.functional.pad(x, (1, 1, 0, 1)))
            x3 = self.conv3_(nn.functional.pad(x, (0, 1, 1, 1)))
            x4 = self.conv4_(nn.functional.pad(x, (0, 1, 0, 1)))
            # print(x1.size(), x2.size(), x3.size(), x4.size())

            x = torch.cat((x1, x2, x3, x4), dim=1)

            x = self.ps(x)
            return x

    class FasterUpProjModule(nn.Module):
        def __init__(self, in_channels):
            super(FasterUpProj.FasterUpProjModule, self).__init__()
            out_channels = in_channels // 2

            self.upper_branch = nn.Sequential(collections.OrderedDict([
                ('faster_upconv', FasterUpProj.faster_upconv(in_channels)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = FasterUpProj.faster_upconv(in_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channel):
        super(FasterUpProj, self).__init__()

        self.layer1 = self.FasterUpProjModule(in_channel)
        self.layer2 = self.FasterUpProjModule(in_channel // 2)
        self.layer3 = self.FasterUpProjModule(in_channel // 4)
        self.layer4 = self.FasterUpProjModule(in_channel // 8)


def choose_decoder(decoder, in_channels):
    if decoder[:6] == 'deconv':
        assert len(decoder) == 7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    elif decoder == "fasterupproj":
        return FasterUpProj(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)

class ResNet(nn.Module):
    def __init__(self, dataset = 'kitti', layers = 50, decoder = 'upproj', output_size=(228, 304), in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)

        self.upSample = choose_decoder(decoder, num_channels // 2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels // 32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)

        self.upSample.apply(weights_init)

        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.conv2(x4)
        x = self.bn2(x)

        x = self.upSample(x)

        x = self.conv3(x)
        x = self.bilinear(x)

        return x

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net layer whose learning rate is 1x lr.
        """
        b = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters of the net layer whose learning rate is 20x lr.
        """
        b = [self.conv2, self.bn2, self.upSample, self.conv3, self.bilinear]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k


class Discriminator(nn.Module):
    """
    Discriminator for FCRN
    """
    def __init__(self, n_in):
        super(Discriminator, self).__init__()
        self.n_in = n_in
        self.modules = []

        self.modules += [   nn.Conv2d(self.n_in, 32, kernel_size=3, stride=(1, 1), padding=(2, 2)),
                            nn.MaxPool2d(kernel_size=(2, 2)),
                            nn.LeakyReLU(0.2, True),
                            nn.Conv2d(32, 32, 3, 1, 1),
                            nn.MaxPool2d(kernel_size=(2,2)),
                            nn.LeakyReLU(0.2, True),
                            nn.Conv2d(32, 32, 3, 1, 1),
                            nn.MaxPool2d(kernel_size=(2,2)),
                            nn.LeakyReLU(0.2, True),
                            nn.Conv2d(32, 32, 3, 1, 1),
                            nn.MaxPool2d(kernel_size=(2,2)),
                            nn.LeakyReLU(0.2, True),
        ]

        self.model = nn.Sequential(*self.modules)
        self.fc = nn.Linear(32 * 4, 1)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)


"""
Define Loss functions here
"""
class ScaleInvariantError(nn.Module):
    """
    Scale invariant error defined in Eigen's paper!
    """

    def __init__(self, lamada=1.0):
        super(ScaleInvariantError, self).__init__()
        self.lamada = lamada
        return

    def forward(self, y_true, y_pred):
        first_log = torch.log(torch.clamp(y_pred, 0, 1))
        second_log = torch.log(torch.clamp(y_true, 0, 1))
        d = first_log - second_log
        loss = torch.mean(d * d) - self.lamada * torch.mean(d) * torch.mean(d)
        return loss


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

    
def set_requires_grad(net, requires_grad=False):
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad

def save_image(model, x, batch, mode="train"):
    pred = model(x)
    pred *= 100
    npimg = pred.cpu().detach().numpy()
    npimg = np.transpose(npimg, (0, 2, 3, 1))
    cv2.imwrite('saved_images/%s_image_%d.jpg'%(mode, batch), npimg[0])


def train(train_loader, val_loader, model, discriminator, criterion_L1, criterion_MSE, 
          criterion_berHu, criterion_GAN, optimizer, optimizer_D, epoch, batch_size, val_frequency=10):
    model.train()  # switch to train mode
    eval_mode = False
    num_batches = 1453 // batch_size
    init_lr = optimizer.param_groups[0]['lr']
    
    total_losses_log = {"l1":0, "mse":0, "berHu":0, "adv":0}
    
    valid_T = torch.ones(batch_size, 1).cuda()
    zeros_T = torch.zeros(batch_size, 1).cuda()
    
    train_loss_list = []
    val_loss_list = []
    
    for iter_ in range(num_batches):
        # Adjust Learning Rate
        if iter_ % 1000 == 0 and optimizer.param_groups[0]['lr'] > init_lr/10.0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.98
        
        input, target = next(train_loader.get_one_batch(batch_size))
        input, target = input.float(), target.float()
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        
        optimizer.zero_grad()
        
        pred = model(input)
        
        loss_L1 = criterion_L1(pred, target)
        loss_MSE = criterion_MSE(pred, target)
        loss_berHu = criterion_berHu(pred, target)
        #loss_SI = criterion_SI(pred, target)
        '''
        set_requires_grad(discriminator, False)
        
        loss_adv = 0
        
        for a in range(3):
            for b in range(9):
                row = 30 * a
                col = 30 * b
                patch_fake = pred[:, :, row:row+30, col:col+30]
                pred_fake = discriminator(patch_fake)
                loss_adv += criterion_GAN(pred_fake, valid_T)
        '''
        
        loss_gen = loss_L1 + loss_berHu
        loss_gen.backward()
        optimizer.step()
        '''
        set_requires_grad(discriminator, True)
        optimizer_D.zero_grad()
        loss_D = 0
        for a in range(3):
            for b in range(9):
                row = 30 * a
                col = 30 * b
                patch_fake = pred[:, :, row:row+30, col:col+30]
                patch_real = target[:, :, row:row+30, col:col+30]
                pred_fake = discriminator(patch_fake.detach())
                pred_real = discriminator(patch_real)
                loss_D_fake = criterion_GAN(pred_fake, zeros_T)
                loss_D_real = criterion_GAN(pred_real, valid_T)
                loss_D += 0.5 * (loss_D_fake + loss_D_real)
        
        loss_D.backward()
        optimizer_D.step()

        torch.cuda.synchronize()
        '''
        total_losses_log["l1"] += loss_L1.item()
        total_losses_log["mse"] += loss_MSE.item()
        total_losses_log["berHu"] += loss_berHu.item()
        #total_losses_log["adv"] += loss_adv.item()
         
        if (iter_ + 1) % 10 == 0:
            save_image(model, input, iter_)
            print('Train Epoch: {} Batch: [{}/{}], L1 ={:0.3f}, MSE={:0.3f}, berHu={:0.3f}'.format(
                epoch, iter_ + 1, num_batches, 
                loss_L1.item(), loss_MSE.item(), loss_berHu.item()))
        
        if (iter_ + 1) % val_frequency == 0:
            val_loss = validate(val_loader, model, discriminator, criterion_L1, criterion_MSE, criterion_berHu, criterion_GAN, batch_size, iter_)
            for key in total_losses_log.keys():
                total_losses_log[key] /= (val_frequency * batch_size)
            train_loss_list.append(total_losses_log)
            val_loss_list.append(val_loss)
            save_losses(train_loss_list, val_loss_list)
            print("TRAIN:- ", total_losses_log)
            print("VAL :- ", val_loss)
            total_losses_log = {"l1":0, "mse":0, "berHu":0, "adv":0}

def save_losses(train_loss_list, val_loss_list):
    out = {"train_loss": train_loss_list, "val_loss":val_loss_list}
    with open("losses.pkl", "wb") as log:
        pickle.dump(out, log)
        
def validate(val_loader, model, discriminator, criterion_L1, criterion_MSE, 
             criterion_berHu, criterion_GAN, batch_size, train_iter):
    model.eval()  # switch to evaluate mode
    #discriminator.eval()
    
    valid_T = torch.ones(batch_size, 1).cuda()
    zeros_T = torch.zeros(batch_size, 1).cuda()
    
    #num_batches = len(val_loader) // batch_size
    total_losses = {"l1":0, "mse":0, "berHu":0, "adv":0}
    num_batches = 25
    for i in range(num_batches):

        input, target = next(val_loader.get_one_batch(batch_size))
        input, target = input.float(), target.float()
        input, target = input.cuda(), target.cuda()
        
        torch.cuda.synchronize()

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
            
        torch.cuda.synchronize()
        
        loss_L1 = criterion_L1(pred, target)
        loss_MSE = criterion_MSE(pred, target)
        loss_berHu = criterion_berHu(pred, target)
        '''
        loss_adv = 0
        with torch.no_grad():
            for a in range(3):
                for b in range(9):
                    row = 30 * a
                    col = 30 * b
                    patch_fake = pred[:, :, row:row+30, col:col+30]
                    pred_fake = discriminator(patch_fake)
                    loss_adv += criterion_GAN(pred_fake, valid_T)
        '''            
        total_losses["l1"] += loss_L1.item()
        total_losses["mse"] += loss_MSE.item()
        total_losses["berHu"] += loss_berHu.item()
        #total_losses["adv"] += loss_adv.item()
        
        if i==0:
            save_image(model, input, train_iter, "val")
    
    for key in total_losses.keys():
        total_losses[key] /= (num_batches * batch_size)
    
    model.train()
    #discriminator.train() 
    return total_losses

raw_data_dir = ""
depth_maps_dir = "depth/"
nyu_dir = "../cnn_depth_tensorflow/data/nyu_datasets/"
print("=> Loading Data ...")
train_loader = DataLoader(nyu_dir)
val_loader = DataLoader(nyu_dir)

print("=> creating Model")
model = ResNet(layers=101, output_size=(90, 270), pretrained=True)
discriminator = Discriminator(3)
discriminator.apply(weights_init)

print("=> model created.")
start_epoch = 0
init_lr = 0.001

train_params = [{'params': model.get_1x_lr_params(), 'lr': init_lr},
                {'params': model.get_10x_lr_params(), 'lr': init_lr * 10}]

optimizer = torch.optim.SGD(train_params, lr=init_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=init_lr)

# You can use DataParallel() whether you use Multi-GPUs or not
model = nn.DataParallel(model).cuda()
#model = model.cuda()
discriminator = discriminator.cuda()

# Define Loss Function
criterion_L1 = MaskedL1Loss()
criterion_berHu = berHuLoss()
criterion_MSE = MaskedMSELoss()
criterion_GAN = nn.BCELoss()

batch_size = 64
val_frequency = 50

for epoch in range(50):
    # Train the Model
    train(train_loader, val_loader, model, discriminator, criterion_L1, criterion_MSE, criterion_berHu, 
          criterion_GAN, optimizer, optimizer_D, epoch, batch_size, val_frequency)
     
    # Save Checkpoint
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './ResNet.pth')

