
import os
from os.path import join
import torch
import logging
import uuid
import json
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from aiohttp import web
from src.videotracks.trained_model import TrainedModelTrack
from src.videotracks.obj_track import ObjVidTrack
from src.obj_seg.fullmodel import ObjectInHand
from src.trainer.models.unet_eff_b0 import SimpleClassifer, JointModel
from collections import OrderedDict


class MyWebRTC:
    def __init__(self, args, vtrack, device, trained_ckpt=None, num_classes=3, joint=False, **kwargs) -> None:
        self.args = args
        self.logger = logging.getLogger("pc")
        self.pcs = set()
        self.relay = MediaRelay()
        self.num_classes = num_classes
        self.vtrackname =vtrack

        if vtrack == "trained":
            self.model = self.init_trained_model(device, ckptpath=trained_ckpt, num_classes=num_classes, joint=joint)
            self.vtrack = TrainedModelTrack(None, model=self.model, transform="vis", jointmodel=joint)
        elif vtrack == "obj":
            self.model =  self.init_obj_seg_model(device)
            self.vtrack = ObjVidTrack(None, self.model, transform="vis", num_classes=num_classes, **kwargs)
        elif vtrack == None:
            self.model = None
            self.vtrack = ObjVidTrack(None, self.model, transform=None, num_classes=num_classes, **kwargs)

    
    def init_obj_seg_model(self, device):
        model = ObjectInHand()
        model.to(device)
        model.load_model(self.args.handckpt, self.args.objckpt)
        model.eval()
        model.warm_up_forward()
        return model

    def init_trained_model(self, device, ckptpath, num_classes=3, joint=False):
        if joint:
            model = JointModel(num_classes=num_classes, dropout=0.2)
        else:
            model = SimpleClassifer(num_classes=num_classes, dropout=0.2)
        def load_remove_module(model, my_state_dict):
            my_new_state_dict = OrderedDict()
            for k, v in my_state_dict.items():
                if k[:7] == "module.":
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                my_new_state_dict[name] = v
            model.load_state_dict(my_new_state_dict)
            return model
        # model.load_state_dict(torch.load(ckptpath)["state_dict"] )
        model = load_remove_module(model, torch.load(ckptpath)["state_dict"])
        model.to(device)
        model.eval()
        return model

    async def offer(self, request, jsonify=False):
        print("vtrackname:", self.vtrackname)
        if jsonify:
            params = request
        else:
            params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        self.pcs.add(pc)

        def log_info(msg, *args):
            self.logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)


        if self.args.record_to:
            recorder = MediaRecorder(self.args.record_to)
        else:
            recorder = MediaBlackhole()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
        
        @pc.on("datachannel")
        def on_datachannel(channel):

            self.vtrack.dc = channel
            print("mounted data channel")
            @channel.on("message")
            async def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send(json.dumps({"pong":message[4:]}))
                elif isinstance(message, str) and message.startswith("class_change"):
                    self.vtrack.classid = int(message.split("_")[-1])-1
                    print(message)

        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)
            # global videotrack
            self.vtrack.track = self.relay.subscribe(track)
            pc.addTrack(self.vtrack)
            if self.args.record_to:
                recorder.addTrack(self.relay.subscribe(track))

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()

        # handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )