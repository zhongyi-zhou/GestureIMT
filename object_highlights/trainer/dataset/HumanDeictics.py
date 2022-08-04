import cv2
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import os
import glob
from os.path import join
import json
import torchvision
# from utils.utils import *
import torchvision.transforms.functional as F

class Hutics(Dataset):
    def __init__(self, name, rootpath, splitjson=None, concat_inputs=True, 
            data_aug=False, norm=True, bgr=False, relpath=False):
        
        self.bgr=bgr
        self.norm = norm
        if relpath:
            self.relpaths = []
        else:
            self.relpaths = None
        self.name = name
        self.concat = concat_inputs
        self.aug = data_aug
        if splitjson is None:
            splitjson = join(rootpath, "train_test_split.json")
        self.splitjson = splitjson
        self.root = rootpath
        self.paths = self.init_path()
    
    def aug_all(self, img, hand, mask):
        if random.random() > 0.5:
            img = F.hflip(img)
            hand =F.hflip(hand)
            mask = F.hflip(mask)
        return img, hand, mask

    def init_path(self):
        pnames = readjson(self.splitjson)[self.name]
        paths = []
        for pname in pnames:
            imgs = glob.glob(join(self.root, pname, "img/*"))
            imgs.sort()
            hands = glob.glob(join(self.root, pname, "hand/*"))
            hands.sort()
            masks = glob.glob(join(self.root, pname, "objmask/*"))
            masks.sort()
            if len(imgs) != len(masks) or len(imgs) !=len(hands):
                raise ValueError(f"find different path list at {imgs}\n {hands} \n {masks}")
            for i in range(len(imgs)):
                paths.append([imgs[i], hands[i], masks[i]])
                if self.relpaths is not None:
                    self.relpaths.append(os.path.relpath(imgs[i], self.root))
        return paths

    def __len__(self):
            return len(self.paths)

    def getimg(self, index):
        imgpath, handpath, maskpath = self.paths[index]
        if self.bgr:
            img = cv2.imread(imgpath)
        else:
            img = cv2.imread(imgpath)[:,:,::-1]
        hand = cv2.imread(handpath, 0)
        mask = cv2.imread(maskpath, 0)
        return img, hand[:, :, np.newaxis], mask[:, :, np.newaxis]

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
            if self.relpaths is None:
                return torch.cat((img, hand), dim=0), mask
            else:
                return torch.cat((img, hand), dim=0), mask, self.relpaths[index]
        elif self.relpaths is None:
            return img, hand, mask
        else:
            return img, self.relpaths[index]

def readjson(filepath):
    with open(filepath, 'r') as json_file:
        mydict = json.load(json_file)
        print(f"load success...!!! at {filepath}")
    return mydict