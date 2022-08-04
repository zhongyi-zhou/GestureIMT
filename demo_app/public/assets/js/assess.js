const WIDTH = 640;
const HEIGHT = 480;
// const IMAGE_INPUT_SIZE = 224;


class Assess {
  constructor() {
    this.timenow = gettimestr();

    this.time_cost = {
      "inference": new AverageMeter(),
      "render": new AverageMeter(),
      "delay": new AverageMeter()
    }

    this.video = document.getElementById("webcam")
    this.testbox = new TestBox("Testing", ["class-1", "class-2", "class-3"]);
    this.webrtc = new WebRTC("webcam", "assess", null)
  
  }


  start() {
    let myself = this
    this.webrtc.start()
    this.webrtc.dc.onmessage = function(evt) {
        let tmp = JSON.parse(evt.data)
        if (tmp["msg"]){
            myself.testbox.update_conf(tmp["msg"])
        }
    }

    this.testbox.backend_sender = (classid) =>{
      console.log("change class to ", classid)
      myself.webrtc.dc.send("class_change_"+classid);
    }
  }

  stop() {
    this.video.pause();
    this.webrtc.stop()
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

  async send_url_param(){
    return uploadStatus(
      {
      "init_by_url_param": this.decode_url_dict()
      }, 
      "assess")
  }
}


window.addEventListener('load', () => {
  mymain = new Assess("webcam");
  mymain.send_url_param().then(()=>{
    mymain.start()
  })

})

$("#play_pause").click(()=> {
  mymain.start_stop()
})


function sum_array(myarr) {
  function add(accumulator, a) {
    return accumulator + a;
  }
  return myarr.reduce(add, 0) 
}
