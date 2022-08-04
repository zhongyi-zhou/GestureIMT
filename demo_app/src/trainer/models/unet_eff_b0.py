import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

def get_light_net(num_classes=3, encoder_weights="imagenet", bgr=True):
    net = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=encoder_weights, in_channels=3, classes=num_classes)
    if bgr and encoder_weights == "imagenet":
        print("use bgr pretrain")
        net.encoder._conv_stem.weight = torch.nn.Parameter(net.encoder._conv_stem.weight[:,[2,1,0],:,:])
    for block in net.decoder.blocks:
        block.conv2 = nn.Identity()
    return net


class SimpleClassifer(nn.Module):
    def __init__(self, num_classes, dropout=0.2, encoder_weights="imagenet", bgr=True) -> None:
        super().__init__()
        self.encoder = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=encoder_weights, in_channels=3, classes=num_classes).encoder
        if bgr and encoder_weights == "imagenet":
            print("use bgr pretrain")
            self.encoder._conv_stem.weight = torch.nn.Parameter(self.encoder._conv_stem.weight[:,[2,1,0],:,:])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, num_classes, bias=False)
        self.num_classes = num_classes
    
    def forward(self, x, show_saliency=False):
        h0,w0 = x.shape[2:]
        x = self.encoder.extract_features(x)
        if show_saliency:
            hh, ww = x.shape[2:]
            bs = x.shape[0]
            # vis = self.fc(x.permute(0,2,3,1).view(-1,1280)).view(1,hh,ww,3).permute(0, 3, 1, 2)
            vis = self.fc(x.permute(0,2,3,1).reshape(-1,1280)).reshape(bs,hh,ww,3).permute(0, 3, 1, 2)
            # x = x.permute(0,2,3,1).reshape(-1,1280)
            # vis = self.fc(x)
            # # print(vis.shape)
            # vis = vis.reshape(bs,hh,ww,3).permute(0, 3, 1, 2)
            vis = F.interpolate(vis, (h0, w0), mode="bilinear", align_corners=True)
        x = F.adaptive_avg_pool2d(x, (1,1))[:,:,0,0]
        x = self.fc(self.dropout(x))
        if not show_saliency:
            return x
        else:
            return x, vis



class JointModel(nn.Module):
    def __init__(self, num_classes, dropout=0.2) -> None:
        super().__init__()
        self.net = get_light_net(num_classes=num_classes+1)
        self.head_classifer = nn.Sequential(
            self.net.encoder._conv_head,
            self.net.encoder._bn1,
            self.net.encoder._swish,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, num_classes, bias=False)

        self.activations = {}
        self.register_hook()
        

    def register_hook(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output[-1]
            return hook
        self.net.encoder.register_forward_hook(get_activation("encoder"))

    def forward(self, x, inc_cam=False):
        h0,w0 = x.shape[2:]
        x = self.net(x)
        logits = self.activations["encoder"]
        logits = self.head_classifer(logits)
        if inc_cam:
            hh, ww = logits.shape[2:]
            cam = self.fc(logits.permute(0,2,3,1).view(-1,1280)).view(1,hh,ww,3).permute(0, 3, 1, 2)
            cam = F.interpolate(cam, (h0, w0), mode="bilinear", align_corners=True)
        logits = F.adaptive_avg_pool2d(logits, (1,1))[:,:,0,0]
        logits = self.dropout(logits)
        logits = self.fc(logits)
        if inc_cam:
            return logits, x, cam
        else:
            return logits, x
