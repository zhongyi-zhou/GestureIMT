import cv2
import numpy as np
import os

def resize_pad(img, size=[512,512]):
    h,w,c = img.shape
    if w>h:
        dw = int(size[0])
        dh = int(dw/w*h)
    else:
        dh = int(size[1])
        dw = int(dh/h*w)
    img_resize = cv2.resize(img, (dw, dh))
    w_pad = size[0] - dw
    h_pad = size[1] - dh
    img_resize_pad = np.pad(img_resize, ((h_pad//2, h_pad//2 + h_pad%2), (w_pad//2, w_pad//2 + w_pad%2), (0,0)))
    return img_resize_pad

def resize_unpad(img, size):
    img_resize = cv2.resize(img, (int(max(size)), int(max(size))))
    if size[0]>size[1]:
        h_unpad = size[0] - size[1]
        return img_resize[h_unpad//2: h_unpad//2+size[1], :]
    else:
        w_unpad = size[1] - size[0]
        return img_resize[:, w_unpad//2: w_unpad//2+size[0]]

import numpy as np
import os
import json
import time


"----------------------------- File I/O -----------------------------"
def readjson(filepath):
    with open(filepath, 'r') as json_file:
        mydict = json.load(json_file)
        print(f"load success...!!! at {filepath}")
    return mydict

def savejson(mydict, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(mydict, json_file)
        print(f"save success...!!! at {filepath}")



"----------------------------- XXXX -----------------------------"
def path_separate(mypath):
    dirname = os.path.dirname(mypath)
    basename, ext = os.path.splitext(os.path.basename(mypath))
    return dirname, basename, ext


def get_name_by_time():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return timestr

def get_name_by_date():
    timestr = time.strftime("%Y%m%d")
    return timestr    

def convert_n_digit(mynumber, digit):
    return str(mynumber).zfill(digit)


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class AverageMeter(object):
    """
    computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class StatsMeter(object):
    """
    computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val_all = []
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        for i in range(n):
            self.val_all.append(val)
        self.avg = self.sum / self.count
        # self.std = np.array(self.val_all).std()
    
    def get_std(self, on_gpu=True):
        if on_gpu:
            return np.array([e.cpu() for e in self.val_all]).std()
        else:
            return np.array(self.val_all).std()
