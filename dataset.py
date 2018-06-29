import numpy as np 
import cv2
from PIL import Image
import os
import pdb
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.io as sio
import math

def trans_int(x):
    return int(math.ceil(x))
color_list = np.array(np.random.random((40, 3)) * 255, int)

def rand_flip(im, gt):
    rand_num = np.random.random()
    if rand_num >= 0.5:
        # im = np.tile(im, [1,1,3])
        im = Image.fromarray(im.astype(np.uint8))
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = np.array(im)

        gt[:, 0] = 1 - gt[:, 0]
    else:
        im = im.astype(np.uint8)
        gt = gt

    return im, gt 

def random_rota(im, gt):
    d = np.random.random()
    d = int((d-0.5)*60)
    # print im.shape
    # im = np.reshape(im, (128,128,1))
    # im = np.tile(im, [1,1,3])
    # print im.shape
    # print im.dtype
    # img = Image.fromarray(im.astype(np.uint8))
    img = Image.fromarray(im)

    img = img.rotate(-d)
    img = np.array(img)
    # img = img[:,:,0]

    gg = np.ones((21,4))
    for i in range(21):
        #if gt[i, 2]==0: continue
        x1 = gt[i, 0]*128
        y1 = gt[i, 1]*128
        x2 = int((x1-64)*np.cos(np.pi/180.0*d) - (y1-64)*np.sin(np.pi/180.0*d))
        x2 += 64
        y2 = int((x1-64)*np.sin(np.pi/180.0*d) + (y1-64)*np.cos(np.pi/180.0*d))
        y2 += 64
        if x2 >= 128 or y2 >= 128 or x2<0 or y2 <0:
            gg[i, 3] = 0
            continue
        gg[i, 0] = x2/128.0
        gg[i, 1] = y2/128.0
        gg[i, 2] = gt[i, 2]
    gg = np.array(gg, dtype=np.float32)

    return img, gg

class MSRA(data.Dataset):
    def __init__(self, imgtxt, datafile, is_training):
        f = open(imgtxt,'r')
        self.data = np.load(datafile)
        self.imgs = f.readlines()
        self.is_training = is_training
        self.horizontalflip = transforms.RandomHorizontalFlip()
        self.colorjitter = transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5)

    def __getitem__(self, index):
        imgname = self.imgs[index].strip('\n')
        img = Image.open('/home1/xzh/msra/' + imgname)
        label = self.data[index]
        img = np.array(img)
        img = np.reshape(img, (128,128,1))
        img = np.tile(img, [1,1,3])
        label = np.reshape(label,(21,3))
        if self.is_training:
            img, label = rand_flip(img, label)
            img = np.reshape(img, (128,128,3))
            img, label = random_rota(img, label)
            img = np.reshape(img, (128,128,3))
            # img = Image.fromarray(img)
            # self.colorjitter(img)
            # img = np.array(img)
        pxy = []
        pl = []
        for i in range(200):
            p = np.random.randint(128,size=2)
            p = list(p)
            # print p, pxy
            while p in pxy:
                p = np.random.randint(128,size=2)
                p = list(p)
            # p = np.array(p)
            pxy.append(p)
            p = np.array(p)
            pz = img[p[0],p[1],0]
            pz = np.reshape(pz,(1,))
            p = np.concatenate((p, pz))
            pl.append(p)
        pl = np.array(pl)
        pl = np.reshape(pl, (200, 3))
        img = np.reshape(img,(128,128,3))
        img = (img / 255.0 - 0.5) / 0.5
        # print img.shape
        img = np.transpose(img,(2,0,1))
        label = np.reshape(label,(21,4))



        return img, label, pl

    def __len__(self):
        assert self.data.shape[0]==len(self.imgs)
        return len(self.imgs)

class MSRA2(data.Dataset):
    def __init__(self, imgtxt, datafile, return_pt, num_point, is_training):
        f = open(imgtxt,'r')
        self.data = np.load(datafile)
        self.imgs = f.readlines()
        self.is_training = is_training
        self.horizontalflip = transforms.RandomHorizontalFlip()
        self.colorjitter = transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5)
        self.return_pt = return_pt
        self.num_point = num_point

    def __getitem__(self, index):
        imgname = self.imgs[index].strip('\n')
        img = Image.open('/home1/xzh/msra/' + imgname)
        label = self.data[index]
        img = np.array(img)
        img = np.reshape(img, (128,128,1))
        img = np.tile(img, [1,1,3])
        label = np.reshape(label,(21,3))
        if self.is_training:
            img, label = rand_flip(img, label)
            img = np.reshape(img, (128,128,3))
            img, label = random_rota(img, label)
            img = np.reshape(img, (128,128,3))
            # img = Image.fromarray(img)
            # self.colorjitter(img)
            # img = np.array(img)
        else:
            oos = np.ones((21,1))
            label = np.concatenate([label, oos], 1)

        im = img[:,:,0]
        ls = np.where(im != 0)
        ptv = im[ls].reshape(-1, 1)
        ls = np.array(ls)
        ls = np.transpose(ls)
        ls = ((ls / 128.0) - 0.5) / 0.5
        ptv = (ptv / 255.0 - 0.5) / 0.5

        pt = np.concatenate([ls, ptv], 1)
        while len(pt) < self.num_point:
            pt = np.concatenate([pt, pt], 0)
        np.random.shuffle(pt)
        pt = pt[:self.num_point]
        # print pt.shape


        pxy = []
        pl = []
        for i in range(200):
            p = np.random.randint(128,size=2)
            p = list(p)
            # print p, pxy
            while p in pxy:
                p = np.random.randint(128,size=2)
                p = list(p)
            # p = np.array(p)
            pxy.append(p)
            p = np.array(p)
            pz = img[p[0],p[1],0]
            pz = np.reshape(pz,(1,))
            p = np.concatenate((p, pz))
            pl.append(p)
        pl = np.array(pl)
        pl = np.reshape(pl, (200, 3))
        img = np.reshape(img,(128,128,3))
        img = (img / 255.0 - 0.5) / 0.5
        # print img.shape
        img = np.transpose(img,(2,0,1))
        label = np.reshape(label,(21,4))



        return img, label, pt

    def __len__(self):
        assert self.data.shape[0]==len(self.imgs)
        return len(self.imgs)

class AFLW(data.Dataset):
    def __init__(self, path_mat, data_root, size_img, is_training):

        data = sio.loadmat(path_mat)
        self.name_list = data['nameList']
        self.kp = data['data']
        self.bbox = np.array(data['bbox'], int)
        self.ra = data['ra']-1
        self.size_img = size_img
        self.data_root = data_root
        self.is_training = is_training

    def __getitem__(self, index):

        if not self.is_training:
            index += 20000
        name_img = self.name_list[self.ra[0, index]]
        # print name_img
        # img = Image.open(self.data_root + name_img[0][0])
        # img = np.array(img)
        # if len(img.shape) == 2: img = np.tile(np.expand_dims(img, -1), (1,1,3))
        # img = img[:,:,[2,1,0]]
        img = cv2.imread(self.data_root + name_img[0][0])
        
        size = img.shape
        size_b = np.array([size[1], size[1], size[0], size[0]], np.float32)

        bb = self.bbox[self.ra[0, index]]
        bb = np.maximum(bb, 0.0)
        bb = np.minimum(bb, size_b)
        bb = np.array(bb, int)
        # print bb
        img_crop = img[bb[2]:bb[3], bb[0]:bb[1], :]
        # print img_crop.shape

        img_crop = cv2.resize(img_crop, (self.size_img, self.size_img), interpolation=cv2.INTER_CUBIC)
        
        ori_size = bb[1] - bb[0]

        anno = self.kp[self.ra[0, index]]

        anno[0: 19] -= bb[0]
        anno[19:] -= bb[2]
        # anno[0: 19] /= (float(bb[1] - bb[0]) / float(self.size_img))
        # anno[19:] /= (float(bb[3] - bb[2]) / float(self.size_img))
        anno[0: 19] /= (float(bb[1] - bb[0]))
        anno[19:] /= (float(bb[3] - bb[2]))


        img_crop = ((img_crop/255.0)-0.5)/0.5

        img_crop = np.transpose(img_crop, (2,0,1))
        return img_crop, anno


    def __len__(self):
        if self.is_training:
            return 20000
        else:
            return 4386


def draw_img(img, cdn):
    size_img = img.shape[0]
    cdn *= size_img

    for i in range (19):
        cv2.circle(img, (trans_int(cdn[i]), trans_int(cdn[i+19])), 2, color_list[i], 2)

def draw_im(img, gt, pred, count, index, LOG_DIR, is_training):
    img = np.transpose(img,(1,2,0))
    img = ((img*0.5 + 0.5) * 255.0).astype(np.uint8)

    size_img = img.shape[0]
    # cdn *= size_img

    img_gt = img.copy()
    img_pred = img.copy()

    draw_img(img_gt, gt)
    draw_img(img_pred, pred)

    #font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
    #cv2.cv.PutText(img_gt, "GT", (30,30), font, (0,255,0))
    #cv2.cv.PutText(img_pred, "PRED", (30,30), font, (0,255,0))

    im_all = np.concatenate((img_gt, img_pred), 1)

    if is_training:
        cv2.imwrite(LOG_DIR + '/train_img/img%06d.png' %(count), im_all)
    else:
        cv2.imwrite(LOG_DIR + '/test_img/img%06d_%02d.png' %(count, index), im_all)



if __name__ == "__main__":
    dataset = AFLW('/data/dataset/face/aflw/AFLWinfo_release.mat', '/data/dataset/face/aflw/data/flickr/', 256, is_training=True)
    train_loader = DataLoader(dataset, batch_size=3, num_workers=1, shuffle=True)
    for i, data in enumerate(train_loader):
        imgs, gt = data
        # print gt.shape
        gt = np.array(gt[0])
        img = imgs[0].numpy()
        # print img
        img = np.transpose(img, (1,2,0)).copy()
        # print img.shape, gt
        # print img
        img = img * 0.5 + 0.5

        img = img * 255.0
        img = np.array(img, np.uint8)
        # print img.min()

        # pdb.set_trace()

        draw_img(img, gt)

        cv2.imshow('img', img)
        cv2.waitKey()


