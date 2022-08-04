import cv2
import numpy as np
import torch

class Vis:
    def __init__(self, img) -> None:
        self.img = img
        self.h, self.w = self.img.shape[:2]
        self.mask = np.zeros((self.h, self.w, 3))
        self.colors = np.array([
            [0, 0, 255], 
            [0, 255, 0],
            [255, 0, 0],
            [1, 190, 200], # orange
        ])
        self.count = 0


    def add_mask(self, mask, scale=255.):
        mask = mask.squeeze()/scale
        # print(mask.shape)
        # print(self.mask.shape)
        self.mask += mask[:,:, np.newaxis] * self.colors[self.count][np.newaxis, np.newaxis,:]
        self.count += 1
        return np.array((0.5*self.img + self.mask*0.5).clip(0, 255), dtype=np.uint8)
    
    

class VisTensor:
    def __init__(self, img, scale=1.0, device="cuda:0") -> None:
        self.img = img.to(device)*(255./scale)
        self.h, self.w = self.img.shape[1:]
        self.mask = torch.zeros(3, self.h, self.w).to(device)
        self.colors = torch.tensor([
            [0, 0, 255], 
            [0, 255, 0],
            [255, 0, 0],
            [1, 190, 200], # orange
        ]).to(device)
        self.count = 0


    def add_mask(self, mask, scale=1., get_val=False):
        mask = mask.squeeze()/scale
        self.mask += mask[None,:,:] * self.colors[self.count][:, None, None]
        self.count += 1
        if get_val:
            return (0.5*self.img + self.mask*0.5).clip(0, 255)
    