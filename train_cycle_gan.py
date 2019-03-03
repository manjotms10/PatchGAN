
import itertools
import os
import pickle
import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from PIL import Image
from matplotlib import pyplot as plt

from nyu_loader import DataLoader

cmap = plt.cm.jet

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def weights_init_normal(m):

    classname = m.__class__.__name__

    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
class EpochTracker():
    def __init__(self, in_file):
        self.epoch = 0
        self.iter = 0
        self.in_file = in_file
        self.file_exists = os.path.isfile(in_file)
        if self.file_exists:
            with open(in_file, 'r') as f:
                d = f.read()
                a, b = d.split(";")
                self.epoch = int(a)
                self.iter = int(b)

    def write(self, epoch, iteration):
        self.epoch = epoch
        self.iter = iteration
        data = "{};{}".format(self.epoch, self.iter)
        with open(self.in_file, 'w') as f:
            f.write(data)

"""
Define Loss functions here
"""
class ScaleInvariantError(nn.Module):
    """
    Scale invariant error defined in Eigen's paper!
    """

    def __init__(self, lamada=0.5):
        super(ScaleInvariantError, self).__init__()
        self.lamada = lamada
        return

    def forward(self, y_true, y_pred):
        first_log = torch.log(torch.clamp(y_pred, 0.00001, 1))
        second_log = torch.log(torch.clamp(y_true, 0.00001, 1))
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

def create_depth_color(depth):
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth = (255 * cmap(depth_relative)[:, :, :3])
    return depth

def save_image(pred_depth, x, y, batch, mode="train"):
    depth = create_depth_color(np.transpose(pred_depth[0].detach().cpu().numpy(), [1,2,0])[:, :, 0])
    target = create_depth_color(np.transpose(y[0].cpu().numpy(), [1,2,0])[:, :, 0])
    orig = 255 * np.transpose(x[0].cpu().numpy(), [1,2,0])

    img = np.concatenate((orig, target, depth), axis =1)
    
    img = Image.fromarray(img.astype('uint8'))
    img.save('saved_images/%s_image_%d.jpg'%(mode, batch))
    
    
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

    
class CycleGanResnetGenerator(nn.Module):
    def __init__(self, in_c, out_c, ngf=32, use_dropout=True):
        super(CycleGanResnetGenerator, self).__init__()

        self.in_channels = in_c
        self.out_channels = out_c
        self.num_resnet_blocks = 9

        model = [nn.ReflectionPad2d(3),
                 SpectralNorm(nn.Conv2d(self.in_channels, ngf, kernel_size=7, padding=0,
                           bias=nn.InstanceNorm2d)),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]

        # we down-sample for 2 layers
        for i in range(2):
            in_ch = 2**i * ngf
            out_ch = 2 * in_ch
            model += [SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2,
                                padding=1, bias=nn.InstanceNorm2d)),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(True)]

        # Add Resnet Blocks
        in_ch = 4 * ngf
        for i in range(self.num_resnet_blocks):
            model += [CycleGanResnetBlock(in_ch, use_dropout)]

        # We up-sample for 2 layers
        for i in range(2):
            in_ch = 2**(2 - i) * ngf
            out_ch = int(in_ch / 2.0)
            model += [SpectralNorm(nn.ConvTranspose2d(in_ch, out_ch,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=nn.InstanceNorm2d)),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [SpectralNorm(nn.Conv2d(ngf, self.out_channels, kernel_size=7, padding=0))]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Resnet Module to be used in Generator
class CycleGanResnetBlock(nn.Module):

    def __init__(self, dim, use_dropout=True):
        super(CycleGanResnetBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                bias=nn.InstanceNorm2d)),
                      nn.BatchNorm2d(dim),
                      nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1),
                       SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                 bias=nn.InstanceNorm2d)),
                       nn.BatchNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class CycleGanDiscriminator(nn.Module):
    def __init__(self, in_c, ndf=32, n_layers=3):
        super(CycleGanDiscriminator, self).__init__()

        self.input_channels = in_c

        model = [SpectralNorm(nn.Conv2d(self.input_channels, ndf,
                      kernel_size=4, stride=2, padding=1)),
                 nn.LeakyReLU(0.2, True)]

        out_ch = ndf

        for n in range(1, n_layers):
            in_ch = out_ch
            out_ch = min(2**n, 8) * ndf
            model += [
                SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2,
                          padding=1, bias=nn.InstanceNorm2d)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True)
            ]

        in_ch = out_ch
        out_ch = min(2**n_layers, 8) * ndf
        model += [
            SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=1, padding=1,
                      bias=nn.InstanceNorm2d)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
        ]

        model += [SpectralNorm(nn.Conv2d(out_ch, 1, kernel_size=4, stride=1, padding=1))]

        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(1064,1)

    def forward(self, input):
        x = self.model(input)
        x = x.view(x.size(0), -1)
        return nn.functional.sigmoid(self.fc(x))


class CycleGAN:

    def __init__(self, device, file_prefix, learning_rate, beta1,
                 train=False):
        print("Starting Cycle Gan with Train = {}".format(train))
        
        self.architecture = 'cycle_gan_'
            
        self.lambda1 = 15.0
        self.lambda2 = 10.0

        self.is_train = train
        self.device = device
        self.file_prefix = file_prefix

        self.epoch_tracker = EpochTracker(file_prefix + self.architecture + "epoch.txt")
        
        self.gen_a_file = file_prefix + self.architecture + 'generator_a.pth'
        self.gen_b_file = file_prefix + self.architecture + 'generator_b.pth'
        self.dis_a_file = file_prefix + self.architecture + 'discriminator_a.pth' 
        self.dis_b_file = file_prefix + self.architecture + 'discriminator_b.pth'

        if self.epoch_tracker.file_exists or not self.is_train:
            self.GenA = self.init_net(CycleGanResnetGenerator(in_c=3, out_c=1), self.gen_a_file)
            self.GenB = self.init_net(CycleGanResnetGenerator(in_c=1, out_c=3), self.gen_b_file)
        else:
            self.GenA = self.init_net(CycleGanResnetGenerator(in_c=3, out_c=1))
            self.GenB = self.init_net(CycleGanResnetGenerator(in_c=1, out_c=3))

        self.real_A = self.real_B = self.fake_A = self.fake_B = self.new_A = self.new_B = None

        if train:
            if self.epoch_tracker.file_exists:
                self.DisA = self.init_net(CycleGanDiscriminator(in_c=1), self.dis_a_file)
                self.DisB = self.init_net(CycleGanDiscriminator(in_c=3), self.dis_b_file)
            else:
                self.DisA = self.init_net(CycleGanDiscriminator(in_c=1))
                self.DisB = self.init_net(CycleGanDiscriminator(in_c=3))

            # define loss functions
            self.criterionGAN = nn.BCELoss()
            self.criterionCycle = nn.L1Loss()
            self.criterionSupervised = nn.L1Loss()
            self.depth_loss = ScaleInvariantError()

            # initialize optimizers
            self.optimizer_g = torch.optim.Adam(itertools.chain(self.GenA.parameters(), self.GenB.parameters()),
                                                lr=learning_rate, betas=(beta1, 0.999))
            self.optimizer_d = torch.optim.Adam(itertools.chain(self.DisA.parameters(), self.DisB.parameters()),
                                                lr=learning_rate, betas=(beta1, 0.999))
            self.optimizers = [self.optimizer_g, self.optimizer_d]

            self.loss_disA = self.loss_disB = self.loss_cycle_A = 0
            self.loss_cycle_B = self.loss_genA = self.loss_genB = 0
            self.supervised_A = self.supervised_B = self.loss_G = 0
        else:
            self.pixelLoss = nn.L1Loss()
            self.test_A = self.test_B = 0

    def set_input(self, real_A, real_B):
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self):
        self.fake_B = self.GenA(self.real_A).to(self.device)
        self.new_A = self.GenB(self.fake_B).to(self.device)

        self.fake_A = self.GenB(self.real_B).to(self.device)
        self.new_B = self.GenA(self.fake_A).to(self.device)

    def backward_d(self, netD, real, fake):
        true = 0.9 * Variable(Tensor(np.ones((real.size(0), 1))), requires_grad=False).to(self.device)
        false = Variable(Tensor(np.zeros((real.size(0), 1))), requires_grad=False).to(self.device)

        predict_real = netD(real)
        loss_d_real = self.criterionGAN(predict_real, true)

        predict_fake = netD(fake.detach())
        loss_d_fake = self.criterionGAN(predict_fake, false)

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()

        return loss_d

    def backward_g(self):
        valid = 0.9 * Variable(Tensor(np.ones((self.real_A.size(0), 1))), requires_grad=False).to(self.device)
        self.loss_genA = self.criterionGAN(self.DisA(self.fake_B), valid)
        self.loss_genB = self.criterionGAN(self.DisB(self.fake_A), valid)

        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.new_A, self.real_A) * self.lambda2
        # Backward cycle loss
        self.loss_cycle_B = self.depth_loss(self.new_B, self.real_B) * self.lambda1

        self.supervised_A = self.depth_loss(self.fake_B, self.real_B) * self.lambda1
        self.supervised_B = self.criterionSupervised(self.fake_A, self.real_A) * self.lambda2

        # combined loss
        self.loss_G = (self.loss_genA + self.loss_genB + self.loss_cycle_A + self.loss_cycle_B)
        self.loss_G += self.supervised_A + self.supervised_B

        self.loss_G.backward()

    def train(self):
        # forward
        self.forward()

        # GenA and GenB
        self.set_requires_grad([self.DisA, self.DisB], False)
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # DisA and DisB
        self.set_requires_grad([self.DisA, self.DisB], True)
        self.optimizer_d.zero_grad()

        # backward Dis A
        self.loss_disA = self.backward_d(self.DisA, self.real_B, self.fake_B)

        # backward Dis B
        self.loss_disB = self.backward_d(self.DisB, self.real_A, self.fake_A)

        self.optimizer_d.step()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.test_A = self.criterionSupervised(self.fake_B, self.real_B)
            self.test_B = self.criterionSupervised(self.fake_A, self.real_A)
    
    def set_eval(self):
        self.GenA.eval()
        self.GenB.eval()
    
    def set_train(self):
        self.GenA.train()
        self.GenB.train()
        
    def save_progress(self, path, epoch, iteration, save_epoch=False):
        path +=  self.architecture 
        
#         img_sample = torch.cat((self.real_A.data, self.fake_A.data, self.real_B.data, self.fake_B.data), -2)
#         save_image(img_sample, path + "{}_{}.png".format(epoch, iteration), nrow=4, normalize=True)

        nets = {self.GenA:self.gen_a_file,
                self.GenB:self.gen_b_file,
                self.DisA:self.dis_a_file,
                self.DisB:self.dis_b_file}

        for net, file in nets.items():
            if save_epoch:
                file = "{}_{}".format(file, epoch)
            if torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), file)
                net.to(self.device)
            else:
                torch.save(net.cpu().state_dict(), file)

        self.epoch_tracker.write(epoch, iteration)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def init_net(self, net, file=None):
        gpu_ids = list(range(torch.cuda.device_count()))
        
        if file is not None:
            epoch = str(self.epoch_tracker.epoch)
            net.load_state_dict(torch.load(file + "_" + epoch))
        else:
            try:
                net.apply(weights_init_normal)
            except:
                pass

        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)

        return net


def train(train_loader, val_loader, cycle_gan, criterion_L1, criterion_MSE, 
          criterion_berHu, criterion_depth, epoch, batch_size, val_frequency):
    
    cycle_gan.set_train()  # switch to train mode
    num_batches = len(train_loader)// batch_size    
    total_losses_log = {"l1":0, "mse":0, "berHu":0, "depth":0}
    if epoch > 0:    
        with open("losses.pkl", "rb") as log:
            out = pickle.load(log)
            train_loss_list = out['train_loss']
            val_loss_list = out['val_loss']
    else:
        train_loss_list = []
        val_loss_list = []
 
    for iter_ in range(num_batches):
        
        input, target = next(train_loader.get_one_batch(batch_size))
        input, target = input.float(), target.float()
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        
        real_A = Variable(input.type(Tensor))
        real_B = Variable(target.type(Tensor))

        cycle_gan.set_input(real_A, real_B)
        cycle_gan.train()
        pred = cycle_gan.fake_B
        
        loss_L1 = criterion_L1(pred, target)
        loss_MSE = criterion_MSE(pred, target)
        loss_berHu = criterion_berHu(pred, target)
        loss_depth = criterion_depth(pred, target)
        
        torch.cuda.synchronize()
        
        total_losses_log["l1"] += loss_L1.item()
        total_losses_log["mse"] += loss_MSE.item()
        total_losses_log["berHu"] += loss_berHu.item()
        total_losses_log["depth"] += loss_depth.item()
          
        if (iter_ + 1) % 20 == 0:
            save_image(pred, input, target, iter_)
            print('Train Epoch: {} Batch: [{}/{}], Depth:{:0.3f} ADV:{:0.3f} L1 ={:0.3f}, MSE={:0.3f}, berHu={:0.3f}, Disc:{:0.3f}'.
                  format(epoch, iter_ + 1, num_batches, loss_depth, cycle_gan.loss_genA.item(),
                loss_L1.item(), loss_MSE.item(), loss_berHu.item(), cycle_gan.loss_disA.item()))
        
        if (iter_ + 1) % val_frequency == 0:
            val_loss = validate(val_loader, cycle_gan, criterion_L1, criterion_MSE, 
                                criterion_berHu, criterion_depth, batch_size, iter_)
            for key in total_losses_log.keys():
                total_losses_log[key] /= val_frequency
            train_loss_list.append(total_losses_log)
            val_loss_list.append(val_loss)
            save_losses(train_loss_list, val_loss_list)
            print("TRAIN:- ", total_losses_log)
            print("VAL :- ", val_loss)
            total_losses_log = {"l1":0, "mse":0, "berHu":0, "depth":0}


def save_losses(train_loss_list, val_loss_list):
    out = {"train_loss": train_loss_list, "val_loss":val_loss_list}
    with open("losses.pkl", "wb") as log:
        pickle.dump(out, log)
        
def validate(val_loader, cycle_gan, criterion_L1, criterion_MSE, 
             criterion_berHu, criterion_depth, batch_size, train_iter):
    cycle_gan.set_eval()
    
    #num_batches = len(val_loader) // batch_size
    total_losses = {"l1":0, "mse":0, "berHu":0, "depth":0}
    num_batches = len(val_loader) // batch_size
    for i in range(num_batches):

        input, target = next(val_loader.get_one_batch(batch_size))
        input, target = input.float(), target.float()
        input, target = input.cuda(), target.cuda()
        
        torch.cuda.synchronize()
        real_A = Variable(input.type(Tensor))
        real_B = Variable(target.type(Tensor))
    
        cycle_gan.set_input(real_A, real_B)
        cycle_gan.test()
        pred = cycle_gan.fake_B
            
        torch.cuda.synchronize()
        with torch.no_grad():
            loss_L1 = criterion_L1(pred, target)
            loss_MSE = criterion_MSE(pred, target)
            loss_berHu = criterion_berHu(pred, target)
            loss_depth = criterion_depth(pred, target)
                    
        total_losses["l1"] += loss_L1.item()
        total_losses["mse"] += loss_MSE.item()
        total_losses["berHu"] += loss_berHu.item()
        total_losses["depth"] += loss_depth.item()
        
        if i==0:
            save_image(pred, input, target, train_iter, "val")
    for key in total_losses.keys():
        total_losses[key] /= num_batches
    cycle_gan.set_train()
    return total_losses

raw_data_dir = ""
depth_maps_dir = "depth/"
nyu_dir = "nyu_datasets/"
print("=> Loading Data ...")
train_loader = DataLoader(nyu_dir)
val_loader = DataLoader(nyu_dir, "val")
project_root = "./"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

models_prefix = project_root + "saved_models/"
images_prefix = project_root + "saved_images/"
ensure_dir(models_prefix)
ensure_dir(images_prefix)

init_lr = 0.001
b1 = 0.5
print("=> creating Model")
cycle_gan = CycleGAN(device, models_prefix, init_lr, b1,train=True)

print("=> model created.")
start_epoch = 0

# Define Loss Function
criterion_L1 = MaskedL1Loss()
criterion_berHu = berHuLoss()
criterion_MSE = MaskedMSELoss()
criterion_depth = ScaleInvariantError()


batch_size = 8
val_frequency = 80
max_epochs = 100

total_batches = int(len(train_loader) / batch_size)

for epoch in range(cycle_gan.epoch_tracker.epoch, max_epochs):
    # Train the Model
    train(train_loader, val_loader, cycle_gan, criterion_L1, criterion_MSE, criterion_berHu, criterion_depth,
          epoch, batch_size, val_frequency)
     
    # Save Checkpoint
    cycle_gan.save_progress(images_prefix, epoch, total_batches, save_epoch=True)