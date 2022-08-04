import os
import sys
from os.path import join
sys.path.append(os.getcwd())
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import draw_segmentation_masks

from src.trainer.models.unet_eff_b0 import JointModel, SimpleClassifer
from src.trainer.dataset.dataset import TeachDataset, MyDataset
import cv2
from src.trainer.utils.utils import *

class Trainer:
    def __init__(self, epoch=25, num_classes=3, jointmodel=True, savedir=None, bgr=True, device="cuda:0", 
        ddp=True, local_rank=0) -> None:
        self.epoch = epoch
        self.bgr = bgr
        self.device = device
        self.local_rank=local_rank
        self.jointmodel = jointmodel
        self.num_classes = num_classes
        self.batch_size = 4
        self.criterion = nn.CrossEntropyLoss()
        if jointmodel:
            self.model = JointModel(num_classes=num_classes, dropout=0.2)
            self.gamma = 1
        else:
            self.model = SimpleClassifer(num_classes=num_classes, dropout=0.2)
        self.model.to(self.device)
        if ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
        
        self.savedir = savedir

    def unnorm(self, img):
        if self.bgr:
            inv_trans = torchvision.transforms.Normalize(
                mean=[-0.406/0.225, -0.456/0.224, -0.485/0.229],
                std=[1/0.225,1/0.224,1/0.229]
            )
        else:
            inv_trans = torchvision.transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229,1/0.224,1/0.225]
            )
        return inv_trans(img)

    def train(self, dataloader, lr=1e-4, writer=None):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(self.epoch):
            for data in dataloader:
                optimizer.zero_grad()
                if self.jointmodel:
                    img, gt, mask = data
                    mask = mask.to(self.device)
                    img = img.to(self.device)
                    gt = gt.to(self.device)   
                    pred_cls, pred_seg = self.model(img)
                    loss = self.criterion(pred_cls, gt) + self.gamma*self.criterion(pred_seg, mask)
                else:
                    img, gt = data
                    img = img.to(self.device)
                    gt = gt.to(self.device)
                    pred_cls = self.model(img)
                    loss = self.criterion(pred_cls, gt)
                loss.backward()
                optimizer.step()
            print(f"epoch: {epoch}, loss: {loss}")
            if writer and self.local_rank == 0:
                img_unnorm = self.unnorm(img)
                img = np.transpose(img_unnorm.detach().cpu().numpy()*255, [0, 2, 3, 1]).astype("uint8")
                writer.add_scalar('loss', loss, epoch)
                writer.add_image(str(epoch) + "_" + '/input', img[0][:,:,::-1].copy(), epoch, dataformats='HWC')
                if self.jointmodel:
                    mask_vis = pred_seg[0].argmax(0).detach().cpu() == torch.arange(1, self.num_classes+1)[:,None,None]
                    mask_vis = draw_segmentation_masks((img_unnorm[0][[2,1,0],:,:].clone().detach().cpu()*255).type(torch.uint8), mask_vis, alpha=0.5)
                    writer.add_image(str(epoch) + "_" + '/pred', np.transpose(mask_vis.numpy(), [1,2,0]), epoch, dataformats='HWC')

        if self.savedir:        
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, join(self.savedir, "model.pt"))
    

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pretraining')
    parser.add_argument("--dataset", required=True, help="for example, ./tmp/000_test/")
    parser.add_argument("--seed", default=1234)
    # parser.add_argument("--logdir", default="trainer/logs/")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    if args.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.local_rank = local_rank
        if args.local_rank == 0:
            print(args)
        torch.distributed.init_process_group(
            backend="nccl", world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print(device)
    else:
        local_rank = 0
    mydataset = TeachDataset(data_dir=args.dataset, include_seg=args.joint, bgr=True, require_norm=True)
    dataloader = MultiEpochsDataLoader(mydataset, batch_size=4, num_workers=8, shuffle=True, pin_memory=True)

    mytrainer = Trainer(epoch=args.epoch, num_classes=3, jointmodel=args.joint, savedir=args.dataset, bgr=True, device=device, ddp=True, local_rank=local_rank)

    mytrainer.train(dataloader, lr=1e-4, writer=SummaryWriter(join(args.dataset, "logs")))
