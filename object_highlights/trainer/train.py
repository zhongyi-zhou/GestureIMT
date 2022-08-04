import os,sys
sys.path.append(os.getcwd())
import time
import torch
import numpy as np
import random
import argparse
from utils.utils import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

from trainer.dataset.HumanDeictics import Hutics
from trainer.dataset.TEgO import TEgO
from models.obj.efficientunet import * 


def train(model, args):
    local_rank = args.local_rank
    log_path = args.logdir
    print(log_path)
    if local_rank == 0:
        writer = SummaryWriter(log_path + '/train')
        writer_val = SummaryWriter(log_path + '/validate')
    else:
        writer, writer_val = None, None

    if args.modelpath:
        modelpath = args.modelpath
        bestpath = os.path.join(path_separate(modelpath), "best.pt")
        args.bestpath = bestpath
    else:
        modelpath = os.path.join(args.logdir, "model.pt")
        bestpath = os.path.join(args.logdir, "best.pt")
        args.bestpath = bestpath
    
    lr_0 = 1e-4
    lr_1 = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0)
    def lambda1(epoch): 
        if epoch< args.epoch/4:
            return 1
        elif epoch < args.epoch*3/4:
            return (lr_1/lr_0) ** ((epoch - args.epoch/4) / (args.epoch/2))
        else:
            return lr_1/lr_0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion = torch.nn.BCEWithLogitsLoss()

    if args.datasetname == "tego":
        dataset_train = TEgO("Training", args.dataset, data_aug=True,norm=True)
    elif args.datasetname == "hutics":
        dataset_train = Hutics("train", args.dataset, data_aug=True,norm=True)
    sampler = DistributedSampler(dataset_train, num_replicas=args.world_size, rank=local_rank, shuffle=True)
    train_data = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=sampler)

    if args.datasetname == "tego":
        dataset_test = TEgO("Testing", args.dataset, concat_inputs=True, norm=True)
    elif args.datasetname == "hutics":
        dataset_test = Hutics("test", args.dataset, concat_inputs=True,norm=True)
    val_data = DataLoader(dataset_test, batch_size=args.batch_size, pin_memory=True, num_workers=8, shuffle=True)

    if local_rank == 0:
        print("train_set_iter: ", train_data.__len__())
        print("val_set_iter: ", val_data.__len__())

    best_score = 0
    best_score = evaluate(args, model, val_data, 0, local_rank, writer_val, best_score=best_score)
    print('training...')
    model.train()
    step = 0
    for epoch in range(args.epoch):
        model.train()
        train_data.sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            optimizer.zero_grad()
            inputs, objmask = data
            inputs = inputs.to(device, non_blocking=True)
            objmask = objmask.to(device, non_blocking=True)
            pred = model(inputs)
            loss = criterion(pred, objmask)
            step += 1
            loss.backward()
            optimizer.step()

            if step % 200 == 1 and local_rank == 0:
                inputs = inputs.cpu().detach().numpy()
                img = inputs[:, :3]
                hand = inputs[:, [3]]
                objmask = objmask.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()
                img = np.transpose(img*255, [0, 2, 3, 1]).astype("uint8")
                hand = np.transpose(hand*255, [0, 2, 3, 1]).astype("uint8")
                objmask = np.transpose(objmask*255, [0, 2, 3, 1]).astype("uint8")
                pred = np.transpose(pred.clip(0,1)*255, [0, 2, 3, 1]).astype("uint8")
                for j in range(1):
                    writer.add_image(str(epoch) + "_" + str(step) + '/input', img[j].copy(), epoch, dataformats='HWC')
                    writer.add_image(str(epoch) + "_" + str(step) + '/hand', hand[j].copy(), epoch, dataformats='HWC')
                    writer.add_image(str(epoch) + "_" + str(step) + '/pred', pred[j].copy(), epoch, dataformats='HWC')
                    writer.add_image(str(epoch) + "_" + str(step) + '/gt', objmask[j].copy(), epoch, dataformats='HWC')
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('learning rate', scheduler.get_last_lr()[0], step)
                print('epoch:{} loss:{:.4e}'.format(epoch, loss))
                model_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }
                torch.save(model_dict, modelpath)

        scheduler.step()
        if epoch % 5 == 1:
            best_score = evaluate(args, model, val_data, epoch, local_rank, writer_val, best_score)
    
    best_score = evaluate(args, model, val_data, epoch, local_rank, writer_val, best_score)
    model_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(model_dict, modelpath)



def evaluate(args, model, val_data, epoch, local_rank, writer, best_score):
    time_stamp = time.time()
    Loss = AverageMeter()
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    IoU = AverageMeter()
    for i, data in enumerate(val_data):
        inputs, objmask = data
        inputs = inputs.to(device, non_blocking=True)
        objmask = objmask.to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(inputs)
        loss = criterion(pred, objmask)
        pred_bi = pred > 0
        IoU.update(torch.logical_and(pred_bi,objmask.bool()).sum() / torch.logical_or(pred_bi,objmask.bool()).sum(),n=objmask.shape[0])
        Loss.update(loss, n=objmask.shape[0])

        if i == 0 and local_rank == 0:
            for j in range(1):
                inputs = inputs.cpu().detach().numpy()
                img = inputs[:, :3]
                hand = inputs[:, [3]]
                objmask = objmask.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()

                img = np.transpose(img*255, [0, 2, 3, 1]).astype("uint8")
                hand = np.transpose(hand*255, [0, 2, 3, 1]).astype("uint8")
                objmask = np.transpose(objmask*255, [0, 2, 3, 1]).astype("uint8")
                pred = np.transpose(pred.clip(0,1)*255, [0, 2, 3, 1]).astype("uint8")
                for j in range(1):
                    writer.add_image(str(epoch) + "_" + str(j) + '/input', img[j].copy(), epoch, dataformats='HWC')
                    writer.add_image(str(epoch) + "_" + str(j) + '/hand', hand[j].copy(), epoch, dataformats='HWC')
                    writer.add_image(str(epoch) + "_" + str(j) + '/pred', pred[j].copy(), epoch, dataformats='HWC')
                    writer.add_image(str(epoch) + "_" + str(j) + '/gt', objmask[j].copy(), epoch, dataformats='HWC')

    eval_time_interval = time.time() - time_stamp

    if local_rank == 0:
        loss = Loss.avg
        iou = IoU.avg
        print('eval time: {}, loss: {:.3f} IoU: {:.3f}'.format(eval_time_interval, loss, iou))
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('iou', iou, epoch)
        if iou > best_score:
            print("Updating best.pt")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                "loss": loss
            }, args.bestpath)
            best_score = iou
        else:
            print("NOT updating best.pt")
            print(f"best iou: {best_score}")
    return best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='minibatch size')
    parser.add_argument('--world_size', default=2, type=int, help='world size')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--datasetname', type=str, required=True, help="tego or hutics")
    parser.add_argument('--logdir', required=False, default=None, help="logdir")
    parser.add_argument('--modelname', default=None, help="")
    parser.add_argument('--modelpath', type=str, default=None, help="model path")
    parser.add_argument('--bestpath', type=str, default=None, help="model path")
    parser.add_argument('--encoder_pretrain', action='store_true', help="finetune with imagenet pretrains")
    # parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank
    if args.local_rank == 0:
        print(args)
    torch.distributed.init_process_group(
        backend="nccl", rank=local_rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.encoder_pretrain:
        print("using imagenet pretrain")
        encoder_weights = "imagenet"
    else:
        print("training from scratch")
        encoder_weights = None
    

    if not args.modelname or args.modelname.split('_')[0]=="unet":
        model = smp.Unet(encoder_name=f"efficientnet-{args.modelname.split('_')[1]}", encoder_weights=encoder_weights, in_channels=4, classes=1)   
    elif args.modelname.split('_')[0] == "unet++":
        model = smp.UnetPlusPlus(encoder_name=f"efficientnet-{args.modelname.split('_')[1]}", encoder_weights=encoder_weights, in_channels=4, classes=1)   
    elif args.modelname.split('_')[0] == "deeplab":
        model = smp.DeepLabV3(encoder_name=f"efficientnet-{args.modelname.split('_')[1]}", encoder_weights=encoder_weights, in_channels=4, classes=1)   
    elif args.modelname.split('_')[0] == "deeplab+":
        model = smp.DeepLabV3Plus(encoder_name=f"efficientnet-{args.modelname.split('_')[1]}", encoder_weights=encoder_weights, in_channels=4, classes=1)     
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if args.logdir is None:
        args.logdir = os.path.join("trainer/logs/", get_name_by_date())

    args.logdir = os.path.join(args.logdir, args.modelname)
    train(model, args)
