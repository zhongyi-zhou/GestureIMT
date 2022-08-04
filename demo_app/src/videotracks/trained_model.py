import json
from os.path import join
import numpy as np
import cv2
from av import VideoFrame
import torch
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from collections import deque
import time
import random
import torch.nn.functional as F
import torchvision

from src.utils.utils import *


class TrainedModelTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, model, transform="vis", jointmodel=False):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.jointmodel = jointmodel
        self.time_step = deque([], maxlen=100)
        self.dc = None
        self.model = model 
        self.classid = 0
        self.save_counter = 0
        self.save_all = False
        self.savedir = join("tmp_img", get_name_by_date())


        
        

    def check_fps(self):
        self.time_step.append(time.time())
        if self.time_step.__len__() == self.time_step.maxlen and random.random()<0.05:
            print(f"fps: {self.time_step.maxlen/(self.time_step[-1] - self.time_step[0])}")

    async def forward(self, img, device, require_norm=True):
        # img = torch.from_numpy(img[:,:,::-1]/255.).permute(2,0,1).float().to(device)
        img = torch.from_numpy(img/255.).permute(2,0,1).float().to(device)
        if (img.shape != (3,480,640)):
            img = F.interpolate(img[None,:,:,:], (480, 640))
        else:
            img = img[None,:,:,:]
        if require_norm:
            # bgr
            trans = torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
            img_norm = trans(img)
        with torch.no_grad():
            if self.jointmodel:
                conf, mask, cam = self.model(img_norm, inc_cam=True)
                cam = cam[:, self.classid]
                cam = (cam - cam.mean())/cam.std()
                mask = (0.718*mask[:, self.classid+1]+0.282*cam).clip(0,1)
            else:
                conf, mask = self.model(img_norm, show_saliency=True)
                mask = mask[:, self.classid]
                mask = (mask - mask.mean())/mask.std()
                mask = mask.clip(0,1)
            if self.dc is not None:
                conf = F.softmax(conf, dim=1)
                self.dc.send(
                    json.dumps({
                        "msg": conf[0].cpu().numpy().tolist(),
                        "time": time.time()
                    })
                )
                await self.dc._RTCDataChannel__transport._data_channel_flush()
                await self.dc._RTCDataChannel__transport._transmit()
            new_frame = ((mask * torch.tensor([0, 1, 0]).cuda()[:, None, None]* 0.5 + img[0]*0.5).clip(0,1)*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        return new_frame



    async def recv(self):
        
        frame = await self.track.recv()
        # print(frame)
        self.check_fps()

        if self.transform == "vis":
            img = frame.to_ndarray(format="bgr24")
            new_frame = await self.forward(img, device="cuda:0")
            new_frame = VideoFrame.from_ndarray(new_frame, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            return new_frame           
        if self.transform == "rgb2bgr":
            img = frame.to_ndarray(format="bgr24")
            if random.random() < 0.05:
                cv2.imwrite("sample.png", img)
            img = img[:,:,::-1]
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame            
        else:
            return frame