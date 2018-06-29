import torch
import numpy as np 
import cv2
import os
from torch.utils.data import DataLoader
import dataset
import argparse
from PIL import Image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn
from torchvision import models
import torch.nn.functional as F
import shutil
import pdb
import sys, time
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, MovingAverage, AverageMeter_Mat

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=100000, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=100000, help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--loss', default='log', help='Loss function [defaultL l2]')
FLAGS = parser.parse_args()


MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir
LOSS = FLAGS.loss

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

NUM_GPU = len(GPU_INDEX.split(','))
BATCH_SIZE = FLAGS.batch_size * NUM_GPU
print 'Batch Size = %d' %(BATCH_SIZE)
name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.mkdir(LOG_DIR + '/train_img')
os.mkdir(LOG_DIR + '/test_img')
os.system('cp %s %s' % (name_file, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
print str(FLAGS)

res50_path = '/home/jacobwang/torch_model/ResNet/resnet50-19c8e357.pth'

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

class resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50, self).__init__()
        #densenet = models.densenet121()
        resnet = models.resnet50()
        if pretrained:
            resnet.load_state_dict(torch.load(res50_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.avg_pool = nn.AvgPool2d(4, stride=4, padding=0)
        self.fc = nn.Linear(2048, 38)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # pdb.set_trace()
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze()        
        x = x.view(-1, 2048)
        x = self.fc(x)
        # x = x.view(BATCH_SIZE, 19, 2)
        x = self.sig(x)

        # pdb.set_trace()

        return x


class l1_loss(nn.Module):
    def __init__(self, with_flag):
        super(l1_loss, self).__init__()
        self.with_flag = with_flag

    def forward(self, pred, gt):
        if self.with_flag:
            loss = torch.mean(torch.abs(pred - gt[:,:,:3]), 2)
            loss = 10 * torch.mean(loss * gt[:, :, 3])
            return loss

        else:
            return torch.mean(torch.abs(pred - gt))


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
    # if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data)
        # nn.init.normal(m.weight.data, mean=0, std=0.01)
        m.bias.data.zero_()

def main():
    train_data = dataset.AFLW('/data/dataset/face/aflw/AFLWinfo_release.mat', '/data/dataset/face/aflw/data/flickr/', 256, is_training=True)
    test_data = dataset.AFLW('/data/dataset/face/aflw/AFLWinfo_release.mat', '/data/dataset/face/aflw/data/flickr/', 256, is_training=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

    # net = model.DenseNet()
    # net = model.DenseNet_nz()
    # net = model1.vgg_19()
    net = resnet50(pretrained=True)
    net.apply(weight_init)


    net.train().cuda()

    loss_func = nn.CrossEntropyLoss().cuda()

    criterion = l1_loss(False).cuda()

    if OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(net.parameters(), BASE_LEARNING_RATE, 0.9, weight_decay=0.0001, nesterov=True)
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), BASE_LEARNING_RATE, weight_decay=0.0001)    ###__0.0001->0.001
    elif OPTIMIZER == 'rmsp':
        optimizer = torch.optim.RMSprop(net.parameters(), BASE_LEARNING_RATE, weight_decay=0.0001)
    # optimizer = torch.optim.SGD(net.parameters(), learning_rate, 0.9, weight_decay=0.0001, nesterov=True)

    count = 1
    epoch = 0


    loss_ma = MovingAverage(100)

    test_loss = AverageMeter()

    while True:
        epoch += 1
        log_string('********Epoch %d********' %(epoch))
        for i, data in enumerate(train_loader):
            imgs, gt = data
            count += 1


            imgs_in = Variable(imgs).float().cuda()
            gt_in = Variable(gt).float().cuda()

            pred = net(imgs_in)
            
            loss = criterion(pred, gt_in)
            # loss = loss0 + loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ma.update(loss.data[0])


            if count % 100 == 0:
                log_string('[Current iter %d, l1 loss is %2.6f, lr: %f]' 
                    %(count, loss_ma.avg, optimizer.param_groups[0]['lr']))
                out_im = imgs[0].numpy()
                out_gt = gt[0].numpy()
                out_pred = pred[0].cpu().data.numpy().squeeze()
                
                dataset.draw_im(out_im, out_gt, out_pred, count, 0, LOG_DIR, is_training=True)
            if count % 1000 == 0:
                validation(test_loader, net, criterion, count, epoch, test_loss)
            if count % 1000 == 0:
                torch.save(net, './' + LOG_DIR + '/' + 'model.pth')
            if count % 15000 == 0 and optimizer.param_groups[0]['lr'] >= (BASE_LEARNING_RATE / 100.0):
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10.0



def validation(dataloader, net, criterion, count, epoch, test_loss):
    log_string('Validating at Epoch %d, iter %d ---------------------------------' %(epoch, count))
    net.eval()
    test_loss.reset()
    time_list = []
    for i, data in enumerate(dataloader):
        imgs, gt = data

        imgs_in = Variable(imgs).float().cuda()
        gt_in = Variable(gt).float().cuda()
        time1 = time.time()
        pred = net(imgs_in)
        time2 = time.time()

        time_list.append(time2-time1)

        loss = criterion(pred, gt_in)
        # pdb.set_trace()
        test_loss.update(loss.data[0])
        if i < 10:
            out_im = imgs[0].numpy()
            out_gt = gt[0].numpy()
            out_pred = pred[0].cpu().data.numpy().squeeze()
            dataset.draw_im(out_im, out_gt, out_pred, count, i, LOG_DIR, is_training=False)
        

    # pdb.set_trace()
    log_string('[Test loss is %2.5f, speed at %0.5f s/frame.]'
        %(test_loss.avg, np.array(time_list).mean()))

    log_string('Validating Done, going back to Training')

    net.train()



if __name__ == "__main__":
    main()