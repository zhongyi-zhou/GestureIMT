function get_url_dict(){
     return Object.fromEntries(new URLSearchParams(location.search));
}

transpose = m => m[0].map((x,i) => m.map(x => x[i]))


function wrap_dict(mydict, key) {
     let newdict = {};
     newdict[key] = mydict;
     return newdict
}

function sleep(milliseconds) {
     const date = Date.now();
     let currentDate = null;
     do {
          currentDate = Date.now();
     } while (currentDate - date < milliseconds);
}


function gettimestr() {
     var date = new Date()
     var datevalues = [
          date.getFullYear(),
          date.getMonth()+1,
          date.getDate(),
          date.getHours(),
          date.getMinutes(),
          date.getSeconds(),
     ];
     return datevalues.join('_')
}


function executeFunctionByName(functionName, context /*, args */) {
     var args = Array.prototype.slice.call(arguments, 2);
     var namespaces = functionName.split(".");
     var func = namespaces.pop();
     for(var i = 0; i < namespaces.length; i++) {
       context = context[namespaces[i]];
     }
     return context[func].apply(context, args);
   }


function VideoFrameToURLs(myframe) {
     const cv=document.createElement("canvas");
     cv.width=myframe.width;
     cv.height=myframe.height;
     const ctx=cv.getContext("2d");
     ctx.drawImage(myframe,0,0);
     return cv.toDataURL();
}


class AverageMeter {
     constructor() {
          this.reset()
     }

     reset() {
          this.val = 0
          this.avg = 0
          this.sum = 0
          this.count = 0
     }

     update(val, n=1) {
          this.val = val
          this.sum += val * n
          this.count += n
          this.avg = this.sum / this.count
     }

}