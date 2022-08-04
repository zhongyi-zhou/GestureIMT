const WIDTH = 640;
const HEIGHT = 480;
// const IMAGE_INPUT_SIZE = 224;


class Teach {
  constructor() {
    // Initiate variables
    this.url_dict = this.decode_url_dict()
    this.offer_type = this.get_offer_type()
    this.timenow = gettimestr();
    this.time_cost = {
      "inference": new AverageMeter(),
      "render": new AverageMeter(),
      "delay": new AverageMeter()
    }
    this.video = document.getElementById("webcam")
    this.trainbox = new TrainBox("Training", "teach");
    this.webrtc = new WebRTC("webcam", "teach", this.offer_type, )
    this.btn_photo = document.getElementById("btn_photo");
    this.upload_intval = 500;
    this.interval = null
    this.cam_btn() 
  }


  start() {
    let self = this
    this.webrtc.start()
    this.webrtc.dc.onmessage = function(evt) {
        let tmp = JSON.parse(evt.data)
        // console.log(tmp)
        if (tmp["command"]){
          // window[tmp["command"]](tmp["args"])
          executeFunctionByName(tmp["command"], self.trainbox, tmp["args"])

        }
    }
    this.trainbox.btn_finish.on("click", ()=>{
      uploadStatus({
        "finish_teach": {}
      }, "teach").then((res) => {
        console.log("response:", res)
        if (res["success"]){alert("upload success!")}
      })
    })
  }

  stop() {
    this.video.pause();
    this.webrtc.stop()
  }

  cam_btn() {
    let self = this
    this.btn_photo.addEventListener('mousedown', () => {
      self.btn_photo.style.backgroundColor = "grey";
      if (this.trainbox.class_id_now == null) {
        console.log("no label specified; quit")
        return
      }
      // this.training = this.trainbox.class_id_now - 1;
      // this.mymodel.sampling = true;
      console.log("mouse down", self.trainbox.class_id_now)
      if (self.offer_type == "auto") {
        self.upload_image()
      }
      else {
        self.interval = setInterval(function() {
          self.upload_image()
        }, self.upload_intval);
      }
      
    });
    this.btn_photo.addEventListener('mouseup', () => 
    {
      self.btn_photo.style.backgroundColor = 'rgb(96, 169, 178)';
      if (self.offer_type != "auto") {
        clearInterval(self.interval)
      }
      // this.training = -1;
      console.log("mouse up", self.trainbox.class_id_now)
      // if(this.mymodel.ready_for_training()) {
      //   this.mymodel.train();
      //   console.log("Finshed training, turn on inference")
      //   this.test_box = new TestBox("Testing", this.trainbox.get_class_names())
      //   this.show_test_divs()
      // } 
      // this.mymodel.sampling = false;
    });
  }

  upload_image() {
    console.log("upload image")
    uploadStatus({
      "trigger_img_saver": {}
    }, "teach").then((res) => {
      // console.log("response:", res)
    })

  }

  decode_url_dict() {
    let decoder = {
      0: "data_only",
      1: "anno_contour",
      2: "anno_click",
      3: "ours",
    }
    let my_url_dict = get_url_dict()
    my_url_dict["interface"] = decoder[my_url_dict["interface"]]
    return my_url_dict
  }

  get_offer_type() {
    if (this.url_dict["interface"] == "ours") return "auto"
    else return "normal"
  }

  async send_url_param(){
    return uploadStatus(
      {
      "init_by_url_param": this.decode_url_dict()
      }, 
      "teach")
  }
  
}


window.addEventListener('load', () => {
  mymain = new Teach("webcam");
  mymain.send_url_param().then(()=>{
    mymain.start()
  })

})


function sum_array(myarr) {
  function add(accumulator, a) {
    return accumulator + a;
  }
  return myarr.reduce(add, 0) 
}
