class TrainBox {
     constructor(divid="Training", wrap_text="teach") {
          this.divid = divid
          this.div = document.getElementById(divid)
          this.stat = {}
          this.max_num_per_class = 0
          this.num_class = 0
          this.class_id_now = null;
          this.backend_sender = null
          this.init()
          this.wrap_text=wrap_text
     }

     init(num_class=3) {
          this.classdiv = $("<div>", {
               id: "all_train_classes",
               class: "",
               
          })
          this.classdiv.appendTo(this.div)
          
          for(let i=1;i<=num_class;i++) {
               this.add_one_class(i)
          }

          let myedit = $("<div>", {
               id: "train_classes_editor",
               class: "add_btn",
          })
          myedit.appendTo(this.div)

          if (this.divid == "Training") {
               let mybutton = $("<button>", {
                    class: "btn",
                    id: "add_one_class",
                    text: "Add",
                    style: "font-size: 24px"
               })
               mybutton.appendTo(myedit)
               $("<i>", {
                    class: "fa fa-plus-circle",
                    "aria-hidden": "true",
                    style: "font-size: 24px"
               }).prependTo(mybutton)
     
               let self = this
               mybutton.on("click", ()=> {
                    console.log("add one class!")
                    this.num_class += 1
                    self.add_one_class(this.num_class)
                    
               })
               $("<br />").appendTo(myedit)

               let div_finish = $("<div>", {
                    id: "train_classes_editor",
                    class: "add_btn",
                    style: "text-align:center"
               })
               div_finish.appendTo(myedit)
               self.btn_finish = $("<button>", {
                    class: "btn btn_confirm",
                    id: "btn_confirm",
                    text: "Finish and Upload",
                    style: "font-size: 24px",
                    
               })
               self.btn_finish.appendTo(div_finish)
          }
          this.num_class = num_class
     }

     add_one_sample(class_id=null) {
          if (class_id==null){class_id=this.class_id_now}
          this.stat[class_id] += 1
          $("div#class_" + class_id + " > p").text(this.stat[class_id])
          let max_changed = false
          if (this.stat[class_id] > this.max_num_per_class) {
               max_changed = true
               this.max_num_per_class = this.stat[class_id]
          }
          this.update_rects(max_changed)
          
     }

     add_one_class(classid) {
          let classname = "class-" + classid.toString()

          let newdiv = $("<div>", {
               id: "class_" + classid,
               class: "train-class-div"
          })
          newdiv.appendTo(this.classdiv)
          
          let self = this
          newdiv.click((event) => {
               self.clear_all_shadow();
               $(event.currentTarget).addClass("train-class-active");
               self.class_id_now = classid;
               uploadStatus({
                    "update_label_active": {classid: classid-1}
               }, self.wrap_text)
          })

          $("<label>", {
               text: classname,
               class: "alignleft control-label",
          }).appendTo(newdiv)

          $("<input>", {
               text: classname,
               class: "alignleft text-input",
               style: "width: 250px",
               css: { 'display': 'none' }
          }).appendTo(newdiv)

          $("<p>", {
               text: 0,
               class: "alignright"
          }).appendTo(newdiv)

          var svg = d3.select(newdiv[0]).append("svg")
          .attr("height", 20)
          .attr("preserveAspectRatio", "xMinYMin meet")
          .attr("viewBox", `0 0 ${newdiv.width()} 20`)

          svg.append('rect')
          .classed('background', true)
          .attr('id', `rect_${classid}_bg`)
          .attr('rx', 10)
          .attr('ry', 10)
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', `${newdiv.width()}`)
          .attr('height', 20)
          .attr('fill', "rgb(230, 230, 230)");

          svg.append('rect')
          .attr('id', `rect_${classid}`)
          .attr('class', 'progress-rect')
          .attr('rx', 10)
          .attr('ry', 10)
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', 20)
          .attr('height', 20)
          .attr('fill', '#60a9b2');

          this.stat[classid] = 0
          this.label_edit()
     }

     update_rects(max_changed) {
          let width_max = parseFloat($(".background").attr("width")) - 20
          if (max_changed) {
               $(".progress-rect").each((i, obj) => {
                    $(obj).attr("width", (20 + this.stat[i+1]/this.max_num_per_class*width_max).toString())
               })
          }
          else {
               $(`#rect_${this.class_id_now}`).each((i, obj) => {
                    let rect_id = $(obj).attr("id").split("_")[1]
                    $(obj).attr("width", (20 + this.stat[rect_id]/this.max_num_per_class*width_max).toString())
               })
          }
     }

     label_edit() {
          $('.control-label').click(function () {
               console.log(this);
               let myinput = this.nextSibling;
               $(this).hide();
               $(myinput).show().focus();
          });
          $('.text-input').keypress(function (event) {
               var keycode = (event.keyCode ? event.keyCode : event.which);
               if(keycode == '13'){
                    let mylabel = this.previousSibling;
                    if (this.value != "") {$(mylabel).text(this.value);}
                    $(this).hide();
                    $(mylabel).show().focus();
               }
          });
          $('.text-input').keyup(function (event) {
               if (event.keyCode == 27) {
                    let mylabel = this.previousSibling;
                    $(this).hide();
                    $(mylabel).show().focus();
               }
          });


     }

     clear_all_shadow() {
          console.log("clear shadows")
          $(".train-class-div").each((i, obj) => {
               $(obj).removeClass("train-class-active")
          })
     }

     get_class_names() {
          let myclassnames = [];
          $(".train-class-div").each((i, obj) => {
               myclassnames.push($(obj).children("label").text());
          })
          return myclassnames
     }
}