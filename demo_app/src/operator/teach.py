import os, sys
from os.path import join
from src.videotracks.webrtc import MyWebRTC
from src.operator.base import Base, BaseOperator


class Teach(Base):
    # conds_all = ["ours", "data_only", "anno_contour", "anno_click"]
    def __init__(self, args, workdir, num_classes=3, device="cuda:0", **kwargs) -> None:
        super().__init__(args, workdir, num_classes, device)
        # self.device=device
        # self.workdir = workdir
        # self.init_workdir(workdir)
        self.webrtc = {
            "auto": MyWebRTC(args, vtrack="obj", device=device, num_classes=num_classes, **kwargs),
            "normal": MyWebRTC(args, vtrack=None, device=device, num_classes=num_classes)
        }
        self.webrtc_id_now = None
        # self.condition = None
        # self.webrtc_teach_auto = MyWebRTC(args, vtrack="obj", device=device, num_classes=num_classes)
        # self.webrtc_teach_normal = MyWebRTC(args, vtrack=None, device=device, num_classes=num_classes)

    def get_webrtc(self):
        if self.condition is None:
            ValueError("expect a specified condition but got None")
        if self.condition == "ours":
            self.webrtc_id_now = "auto"
            return self.webrtc[self.webrtc_id_now]
        else:
            self.webrtc_id_now = "normal"
            return self.webrtc[self.webrtc_id_now]

    def update_condition(self, condition):
        assert condition in self.conds_all, f"expect condition in {self.conds_all} but got {condition}"
        self.condition = condition
        self.get_webrtc().vtrack.datasetpath = join(self.workdir, self.condition, "dataset")
    
    # def update_workdir(self, workdir):
    #     self.workdir = workdir
    #     self.init_workdir(workdir)

    def add_num_classes(self):
        self.num_classes += 1

    def reinit_vtrack(self):
        self.get_webrtc().vtrack.reinit_counter()
    # def init_workdir(self, workdir):
    #     for name in self.conds_all:
    #         self.init_one_condition(workdir, name)

    # def init_one_condition(self, workdir, condname):
    #     os.makedirs(join(workdir, condname, "dataset"), exist_ok=True)
    #     os.makedirs(join(workdir, condname, "logs/train"), exist_ok=True)

    def trigger_data_saver(self):
        # print("trigger")
        self.get_webrtc().vtrack.save_trigger = True
    
    # def update_label(self, label):
    #     self.webrtc[self.webrtc_id_now].vtrack.label = label


    # def save_data(self, img, label, mask=None ,norm=True, bgr=False):
    #     # img [h, w, 3]
    #     # label: (1,)
    #     if not self.joint:
    #         self.dataset.add_data([img, label], require_norm=norm, bgr=bgr)
    #     else:
    #         self.dataset.add_data([img, label, mask], require_norm=norm, bgr=bgr)

class TeachOperator(BaseOperator):
    def __init__(self, args, teacher) -> None:
        super().__init__(args, actname="teacher", assess=None, teacher=teacher)
        self.teacher = teacher
        self.args = args

    
    def trigger_img_saver(self):
        if self.teacher.get_webrtc().vtrack.save_trigger:
            print("img saver has already been triggered!")
            raise Warning("the number of figures saved may be fewer than expected") 
        else:
            self.teacher.get_webrtc().vtrack.save_trigger = True
            self.teacher.get_webrtc().vtrack.label_cache = self.teacher.get_webrtc().vtrack.label
            # saved = False
            # for i in range(200):
            #     if self.teacher.get_webrtc().vtrack.save_trigger is False:
            #         return True
            #     time.sleep(0.01)
            # return False
    
    def finish_teach(self,): 
        savepath = join(self.teacher.workdir, self.teacher.condition, "label.txt")
        print(self.teacher.get_webrtc().vtrack.label_list)
        with open(savepath, "w") as f:
            for i, line in enumerate(self.teacher.get_webrtc().vtrack.label_list):
                f.writelines([f"{i}, {line}\n"])
        print(f"successfully saved label at {savepath}")
    
    def clear_label(self,): 
        # savepath = join(self.teacher.workdir, self.teacher.condition, "label.txt")
        # print(self.teacher.get_webrtc().vtrack.label_list)
        # with open(savepath, "w") as f:
        #     for i, line in enumerate(self.teacher.get_webrtc().vtrack.label_list):
        #         f.writelines([f"{i}, {line}\n"])
        self.teacher.get_webrtc().vtrack.label_list = []
        print(f"successfully cleared label!")
            
        # self.teacher.get_webrtc().vtrack.label = classid
    
    def init_by_url_param(self, user, interface):
        self.actor.update_workdir(join(self.args.dataroot, user))
        # join(self.args.dataroot, user, interface)
        self.actor.update_condition(interface)
        self.actor.reinit_vtrack()
    # def update_label_active(self, classid):
    #     print(f"updating active label to {classid}")
    #     self.teacher.get_webrtc().vtrack.label = classid


    # def init_by_url_param(self, user, interface):
    #     self.teacher.update_workdir(join(self.args.dataroot, user))
    #     # join(self.args.dataroot, user, interface)
    #     self.teacher.update_condition(interface)

    # async def get_offer(self, request, mydict=None):
    #     if mydict is None and request.body_exists:
    #         mydict = await request.json()
    #     keyname = mydict["offer_type"]
    #     print(keyname)
    #     # return self.teacher.get_webrtc().offer
    #     myval = await self.teacher.webrtc[keyname].offer(request)
    #     # print(myval)
    #     return myval