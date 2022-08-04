class TestBox {
     constructor(divid, classnames) {
          this.divid = divid
          this.div = document.getElementById(divid)
          this.stat = {}
          this.max_num_per_class = 0
          this.num_class = classnames.length
          this.class_id_now = null;
          this.init(classnames);
          this.backend_sender = null;
     }

     init(classnames) {
          this.classdiv = $("<div>", {
               id: "all_test_classes",
               class: "",
               
          })
          this.classdiv.appendTo(this.div)
          
          for(let i=1;i<=this.num_class;i++) {
               this.add_one_class(i, classnames[i-1])
          }
          this.label_edit()
     }

     add_one_sample() {
          this.stat[this.class_id_now] += 1
          $("div#class_" + this.class_id_now + " > p").text(this.stat[this.class_id_now])
          let max_changed = false
          if (this.stat[this.class_id_now] > this.max_num_per_class) {
               max_changed = true
               this.max_num_per_class = this.stat[this.class_id_now]
          }
          this.update_rects(max_changed)
     }

     update_conf(confs) {
          let width_max = parseFloat($(".background-test").attr("width")) - 20
          $(".progress-rect-test").each((i, obj) => {
               let mywidth = (20 + confs[i]*width_max).toString();
               $(obj).attr("width", mywidth)
               $(obj).attr("fill", d3.interpolateGnBu(confs[i]))
               obj.parentElement.previousSibling.innerHTML = (confs[i]*100).toFixed(1)
          })
     }

     add_one_class(classid, classname=null) {
          if (classname == null) {classname = "class" + classid.toString()} 

          let newdiv = $("<div>", {
               id: "testclass_" + classid,
               class: "train-class-div"
          })
          newdiv.appendTo(this.classdiv)
          
          let myobj = this
          newdiv.click((event) => {
               myobj.clear_all_shadow();
               if (myobj.class_id_now == null) {
                    $(event.currentTarget).addClass("train-class-active");
                    myobj.class_id_now = classid;
               }
               else if ($(event.currentTarget).attr("id").split("_")[1] == myobj.class_id_now.toString()) {
                    $(event.currentTarget).removeClass("train-class-active");
                    myobj.class_id_now = null;
               }
               else {
                    $(event.currentTarget).addClass("train-class-active");
                    myobj.class_id_now = classid;
               }
               if (myobj.backend_sender != null){
                    console.log("trigger backend");
                    myobj.backend_sender(classid)
               }
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
          .classed('background-test', true)
          .attr('id', `rect_${classid}_bg_test`)
          .attr('rx', 10)
          .attr('ry', 10)
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', `${newdiv.width()}`)
          .attr('height', 20)
          .attr('fill', "rgb(230, 230, 230)");

          svg.append('rect')
          .attr('id', `rect_${classid}_test`)
          .attr('class', 'progress-rect-test')
          .attr('rx', 10)
          .attr('ry', 10)
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', 20)
          .attr('height', 20)
          .attr('fill', '#60a9b2');

          this.stat[classid] = 0
     }

     update_rects(max_changed) {
          let width_max = parseFloat($(".background-test").attr("width")) - 20
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
}