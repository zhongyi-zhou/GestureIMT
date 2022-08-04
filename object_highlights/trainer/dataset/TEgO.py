import cv2
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import os
import glob
from os.path import join
import torchvision.transforms.functional as F
import torchvision


class TEgO(Dataset):
    def __init__(self, name, rootpath, data_aug=True, norm=True, concat_inputs=True, bgr=False, relpath=False, h=640, w=480):
        self.bgr = bgr
        self.aug = data_aug
        self.norm = norm
        self.concat = concat_inputs
        self.path = os.path.join(rootpath, name)
        self.dataset_name = name
        self.relpath = relpath
        self.relpaths = self.init_datapath()
        print(f"{self.relpaths.__len__()} samples in total.")
        self.h = int(h)
        self.w = int(w)
        
        print("height:", h, "width:", w)

    def __len__(self):
        return len(self.imgpaths)

    def aug_all(self, img, hand, mask):

        if random.random() > 0.5:
            img = F.hflip(img)
            hand =F.hflip(hand)
            mask = F.hflip(mask)
        return img, hand, mask

    def init_datapath(self, with_hand=True):
        self.pathdirs = {
            "img": os.path.join(self.path, "Images"),
            "handmask": os.path.join(self.path, "Masks"),
            "objmask": os.path.join(self.path, "ObjectMask"),
        }
        relpaths = []
        dir = self.pathdirs["img"]
        for subdir in os.listdir(dir):
            for imgdirname in os.listdir(os.path.join(dir, subdir)):
                if with_hand and imgdirname.split("_")[2] == "NH":
                    continue
                tmppaths = glob.glob(os.path.join(dir, subdir, imgdirname, "*.jpg"))
                relpaths += [os.path.relpath(mypath, dir) for mypath in tmppaths]
        self.imgpaths = [join(self.pathdirs["img"], relpath) for relpath in relpaths]
        self.handpaths = [join(self.pathdirs["handmask"], relpath) for relpath in relpaths]
        self.objpaths = [join(self.pathdirs["objmask"], relpath) for relpath in relpaths]
        return relpaths


    def getimg(self, index):
        if self.bgr:
            img = cv2.imread(self.imgpaths[index])
        else:
            img = cv2.imread(self.imgpaths[index])[:,:,::-1]
        handmask = cv2.imread(self.handpaths[index], 0)
        th, handmask = cv2.threshold(handmask, 30, 255, cv2.THRESH_BINARY)
        objmask = cv2.imread(self.objpaths[index], 0)
        if img.shape != (self.h, self.w, 3):
            img = cv2.resize(img, (self.w, self.h))
            handmask = cv2.resize(handmask, (self.w, self.h))
            objmask = cv2.resize(objmask, (self.w, self.h))
        return img, handmask[:, :, np.newaxis], objmask[:, :, np.newaxis]

    def __getitem__(self, index):
        img, hand, mask = self.getimg(index)
        img = torch.from_numpy(img.copy()).permute(2, 0, 1) / 255.
        hand = torch.from_numpy(hand.copy()).permute(2, 0, 1) / 255.
        mask = torch.from_numpy(mask.copy()).permute(2, 0, 1) / 255.

        if self.norm:
            if self.bgr:
                # bgr
                trans = torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])                
            else:
                # rgb
                trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img = trans(img)

        if self.aug:
            img, hand, mask = self.aug_all(img, hand, mask)
        
        if self.concat:
            if self.relpath:
                return torch.cat((img, hand), dim=0), mask
            else:
                return torch.cat((img, hand), dim=0), mask, self.relpaths[index]
        return img, hand, mask
        