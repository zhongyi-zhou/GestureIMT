
import os, sys
from os.path import join
from src.videotracks.webrtc import MyWebRTC
from src.operator.base import Base, BaseOperator

class Assess(Base):

    def __init__(self, args, workdir, num_classes=3, device="cuda:0") -> None:
        super().__init__(args, workdir, num_classes, device)
        self.webrtc_id_now = None
    
    def init_webrtc(self, joint=False):
        if self.condition is None:
            raise ValueError("Please initialize the condition before init webrtc")
        self.webrtc = MyWebRTC(self.args, vtrack="trained", device=self.device, 
            trained_ckpt=join(self.workdir, self.condition, "model.pt"),num_classes=self.num_classes, joint=joint)
        
    def get_webrtc(self):
        if self.webrtc is not None:
            return self.webrtc
        else:
            raise ValueError("Please initialize the condition before use webrtc")
        

class AssessOperator(BaseOperator):
    def __init__(self, args, assess) -> None:
        super().__init__(args, actname="assess", assess=assess)
        self.assess = assess
        self.args = args
 

    def init_by_url_param(self, user, interface):
        self.actor.update_workdir(join(self.args.dataroot, user))
        self.actor.update_condition(interface)
        if interface == "data_only":
            joint=False
        else:
            joint=True
        self.assess.init_webrtc(joint)
