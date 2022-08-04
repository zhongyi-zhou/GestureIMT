from src.videotracks.webrtc import MyWebRTC
import os, sys
from os.path import join


class Base:
    conds_all = ["ours", "data_only", "anno_contour", "anno_click"]
    def __init__(self, args, workdir, num_classes=3, device="cuda:0") -> None:
        self.device=device
        self.workdir = workdir
        self.args = args
        self.num_classes = num_classes
        self.condition = None
    
    def update_condition(self, condition):
        assert condition in self.conds_all, f"expect condition in {self.conds_all} but got {condition}"
        self.condition = condition
        # self.get_webrtc().vtrack.datasetpath = join(self.workdir, self.condition, "dataset")
    
    def update_workdir(self, workdir):
        self.workdir = workdir
        self.init_workdir(workdir)

    def init_workdir(self, workdir):
        for name in self.conds_all:
            self.init_one_condition(workdir, name)

    def init_one_condition(self, workdir, condname):
        os.makedirs(join(workdir, condname, "dataset"), exist_ok=True)
        os.makedirs(join(workdir, condname, "logs/train"), exist_ok=True)
    
    def update_label(self, label):
        self.webrtc[self.webrtc_id_now].vtrack.label = label
    
    

class BaseOperator:
    def __init__(self, args, actname, assess=None, teacher=None, ) -> None:
        self.args = args
        self.teacher = teacher
        self.assess = assess
        # self.actor = actor
        self.update_active(actname)   

    def update_active(self, actname):
        if actname == "teacher":
            self.actor = self.teacher
        elif actname == "assess":
            self.actor = self.assess
        else:
            raise Warning(f"unknown actor: {actname}")

    def update_status(self, stat_dict):
        if stat_dict.keys().__len__() != 1:
            raise ValueError("please wrap the function name with args")
        funcname = list(stat_dict.keys())[0]
        return getattr(self, funcname)(**stat_dict[funcname])
    
    async def get_offer(self, request, mydict):
        if mydict is None and request.body_exists:
            mydict = await request.json()
        keyname = mydict["offer_type"]
        # print(keyname)
        if keyname is None:
            myval = await self.actor.webrtc.offer(request)
        else:
            myval = await self.actor.webrtc[keyname].offer(request)
        print(myval)
        return myval
    
    def init_by_url_param(self, user, interface):
        self.actor.update_workdir(join(self.args.dataroot, user))
        # join(self.args.dataroot, user, interface)
        self.actor.update_condition(interface)
    
    def update_label_active(self, classid):
        print(f"updating active label to {classid}")
        self.actor.get_webrtc().vtrack.label = classid