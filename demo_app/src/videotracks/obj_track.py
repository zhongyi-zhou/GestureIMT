import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
from os.path import join
import numpy as np
import cv2
from aiohttp import web
from av import VideoFrame
import torch
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from collections import deque
from queue import Queue
import time
import random
from src.obj_seg.fullmodel import ObjectInHand
from src.utils.utils import *


class ObjVidTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"
    # track, model, transform="vis", jointmodel=False)
    # def __init__(self, track, transform, args, num_classes=3, model=None):
    def __init__(self, track, model, transform="vis", num_classes=3, saveroot=None):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        if transform not in ["vis", "rgb2bgr", None]:
            raise ValueError(f"Unknown transform: {transform}; expected vis, rgb2bgr or None")
        self.time_step = deque([], maxlen=100)
        self.model = model
        # self.caches = None
        # self.caches_ready = True
        self.dc = None
        self.label = None
        self.label_cache = None
        self.label_list = []
        self.counter = 0
        # self.counter = [0] * num_classes
        print("saveroot:", saveroot)
        self.saveroot = saveroot
        if self.saveroot is not None:
            if not os.path.exists(self.saveroot):
                os.makedirs(self.saveroot, exist_ok=True)
        self.save_trigger = False
        self.save_counter = 0
    
        
    def reinit_counter(self,):
        print("reinit counter!")
        self.label_list = []
        self.counter = 0
        

    def check_fps(self):
        self.time_step.append(time.time())
        if self.time_step.__len__() == self.time_step.maxlen and random.random()<0.005:
            print(f"fps: {self.time_step.maxlen/(self.time_step[-1] - self.time_step[0])}")

    async def save_data(self, img, mask=None, is_tensor=False, bgr=True):
        # print("saving data")
        if self.datasetpath is None:
            raise ValueError("please specify the dataset root path before saving the data")
        if is_tensor:
            img = (img.cpu().permute(1,2,0).numpy()*255).astype("uint8")
        if not bgr:
            img = img[:,:,::-1]
        savepath = join(self.datasetpath, "img", f"{convert_n_digit(self.counter, digit=4)}.jpg")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        cv2.imwrite(savepath, img)
        if mask is not None:
            if is_tensor:
                mask = (mask.cpu().permute(1,2,0).numpy()*255).astype("uint8")
            savepath = join(self.datasetpath, "mask", f"{convert_n_digit(self.counter, digit=4)}.png")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            cv2.imwrite(savepath, mask)
        self.label_list.append(self.label_cache)
        # print("finish saving data")
        print(self.counter)
        if self.dc is not None:
            self.dc.send(
                json.dumps({
                    "command": "add_one_sample",
                    "args": self.label_cache+1
                })
            )
            await self.dc._RTCDataChannel__transport._data_channel_flush()
            await self.dc._RTCDataChannel__transport._transmit()
        self.counter += 1

    async def recv(self):
        
        frame = await self.track.recv()
        self.check_fps()

        if self.transform == "vis":
            img = frame.to_ndarray(format="bgr24")
            # img = torch.from_numpy(img[:,:,::-1]/255.).permute(2,0,1).float().cuda()
            img = torch.from_numpy(img/255.).permute(2,0,1).float().cuda()
            if (img.shape != (3,480,640)):
                img = torch.nn.functional.interpolate(img[None,:,:,:], (480, 640))[0]
            with torch.no_grad():
                mask, _ = self.model(img[None, :,:,:])
            new_frame = ((mask[0] * torch.tensor([0, 0, 1]).cuda()[:, None, None]*0.5 + img*0.5).clip(0,1)*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            # if self.label_cache is not None and self.save_trigger:
            if self.save_trigger:
                await self.save_data(img, mask=mask[0], is_tensor=True, bgr=True)
                self.save_trigger = False
                # tmp_img = ((mask[0] * torch.tensor([1, 1, 1]).cuda()[:, None, None]).clip(0,1)*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                # print(join(self.saveroot, f"{convert_n_digit(self.save_counter, 4)}_mask.jpg"))
                # cv2.imwrite(join(self.saveroot, f"{convert_n_digit(self.save_counter, 4)}_mask.jpg"), tmp_img)
                # tmp_img = (img.clip(0,1)*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                # cv2.imwrite(join(self.saveroot, f"{convert_n_digit(self.save_counter, 4)}_img.jpg"), tmp_img)
                # # myimg = ((mask * torch.tensor([0, 1, 0]).cuda()[:, None, None]* 0.5 + img[0]*0.5).clip(0,1)*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                # cv2.imwrite(join(self.saveroot, f"{convert_n_digit(self.save_counter, 4)}_vis.jpg"), new_frame)
                # self.save_counter += 1
                
            new_frame = VideoFrame.from_ndarray(new_frame, format="bgr24")
            # new_frame =  VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame           
        elif self.transform == "rgb2bgr":
            img = frame.to_ndarray(format="bgr24")
            if random.random() < 0.05:
                cv2.imwrite("sample.png", img)
            img = img[:,:,::-1]
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame            
        elif self.transform == None:
            img = frame.to_ndarray(format="bgr24")
            if (img.shape != (480,640, 3)):
                cv2.resize(img, (640, 480))
            if self.label_cache is not None and self.save_trigger:
                await self.save_data(img, mask=None, is_tensor=False, bgr=True)
                self.save_trigger = False
                # self.update_caches_numpy(img[:,:,::-1], mask=None)
            return frame
