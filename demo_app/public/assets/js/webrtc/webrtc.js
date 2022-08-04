
class WebRTC{
    constructor(vidid, stage, offer_type){
        this.stage = stage
        this.offer_type = offer_type
        this.vidid = vidid
        this.dcInterval = null
        this.video = document.getElementById("webcam")
        this.pc = this.createPeerConnection(this.vidid);
        this.dc = this.create_dc()
    }

    createPeerConnection(vidid) {
        var config = {
            sdpSemantics: 'unified-plan'
        };
    
        let pc = new RTCPeerConnection(config);
    
        pc.addEventListener('track', function(evt) {
            console.log(evt)
            if (evt.track.kind == 'video'){
                document.getElementById("webcam").srcObject = evt.streams[0];
            }
        });
    
        return pc;
    }
    
    create_dc() {
        var time_start = null;
        function current_stamp() {
            if (time_start === null) {
                time_start = new Date().getTime();
                return 0;
            } else {
                return new Date().getTime() - time_start;
            }
        }
        var parameters = {}
        let dc = this.pc.createDataChannel('chat', parameters);
        dc.onclose = function() {
            clearInterval(this.dcInterval);
        };
        dc.onopen = function() {
            var message = 'ping ' + current_stamp();
            dc.send(message);
            console.log("data channel open")
        };
        dc.onmessage = function(evt) {
            if (Math.random() < 0.01) {
                console.log("data channel receives info")       
                console.log(JSON.parse(evt.data))            
            }
        };
        return dc
    }
    

    
    negotiate() {
        let self = this
        let pc = this.pc
        return pc.createOffer().then(function(offer) {
            return pc.setLocalDescription(offer);
        }).then(function() {
            return new Promise(function(resolve) {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    function checkState() {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    }
                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });
        }).then(function() {
            var offer = pc.localDescription;
            return fetch("/offer", {
                body: JSON.stringify({
                    stage: self.stage,
                    offer_type: self.offer_type,
                    sdp: offer.sdp,
                    type: offer.type,
                    video_transform: "vis"
                }),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST'
            });
        }).then(function(response) {
            var myjson = response.json()
            console.log(myjson)
            return myjson;
        }).then(function(answer) {
            console.log(answer)
            self.video.play();
            return pc.setRemoteDescription(answer);
        }).catch(function(e) {
            alert(e);
        });
    }

    start() {
        self = this
        var constraints = {
            video: {
                width: 640,
                height: 480, 
                frameRate: {
                    min: 20,  // very important to define min value here
                    ideal: 24,
                    max: 25,
                },
            },
            audio: false
        };
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            console.log(stream)
            stream.getTracks().forEach(function(track) {
                self.pc.addTrack(track, stream);
            });
            return self.negotiate();
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
    }
    
    stop() {
        document.getElementById('stop').style.display = 'none';
        if (this.dc) {
            this.dc.close();
        }
    
        // close transceivers
        if (this.pc.getTransceivers) {
            this.pc.getTransceivers().forEach(function(transceiver) {
                if (transceiver.stop) {
                    transceiver.stop();
                }
            });
        }
    
        // close local audio / video
        this.pc.getSenders().forEach(function(sender) {
            sender.track.stop();
        });
    
        // close peer connection
        setTimeout(function() {
            this.pc.close();
        }, 500);
    }
}
