import argparse
import asyncio
import json
import logging
import os,sys
import ssl
from os.path import join
from aiohttp import web
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
import json
from src.utils.utils import *
from src.utils.web_handler import img_decoder
from src.videotracks.webrtc import MyWebRTC
from src.operator.teach import Teach, TeachOperator
from src.operator.assess import Assess, AssessOperator

ROOT = os.path.dirname(__file__)
STATIC_PATH = os.path.join(ROOT, "public")    
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

videotrack = None


async def index_teach(request):
    content = open(os.path.join(ROOT, "public", "teach.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def index_assess(request):
    content = open(os.path.join(ROOT, "public", "assess.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

mime = {
    "html": 'text/html',
    "txt": 'text/plain',
    "css": 'text/css',
    "gif": 'image/gif',
    "jpg": 'image/jpeg',
    "png": 'image/png',
    "svg": 'image/svg+xml',
    "js": 'application/javascript'
}


async def upload_file(request):
    # print(request.__dir__())
    if request.body_exists:
        mydict = await request.json()
    img = img_decoder(mydict=mydict, savepath="tmp/img.jpg")

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"success": True}
        ),
    )

async def update_status(request):
    if request.body_exists:
        mydict = await request.json()
    if mydict.keys().__len__() != 1:
        raise Warning(f"Expect one warper from the frontend, but got {mydict.keys()}")
    key = list(mydict.keys())[0]
    assert key in ["teach", "assess"], f"expect either 'teach' or 'assess', but got {key}"
    if key == "teach":
        response = opt_teach.update_status(mydict[key])
    elif key == "assess":
        response = opt_assess.update_status(mydict[key])
    else:
        print("Not supported yet")

    if response is not False:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"success": True}
            ),
        )
    else:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"success": False}
            ),
        )


async def get_offer(request):
    if request.body_exists:
        mydict = await request.json()
    key = mydict["stage"]
    print("stage key:", key)
    if key == "teach":
        resp = await opt_teach.get_offer(request=request, mydict=mydict)
    elif key == "assess":
        resp = await opt_assess.get_offer(request=request, mydict=mydict)
    return resp



parser = argparse.ArgumentParser(
    description="WebRTC audio / video / data-channels demo"
)
parser.add_argument("--cert-file", default="cert.pem", help="SSL certificate file (for HTTPS)")
parser.add_argument("--key-file", default="key.pem", help="SSL key file (for HTTPS)")
parser.add_argument(
    "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
)
parser.add_argument(
    "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
)
parser.add_argument('--handckpt', default="src/ckpt/resnet18_adam.pth.tar", type=str,
                        help="hand segmentation model path")
parser.add_argument('--objckpt', default="src/ckpt/unet-b0-bgr-100epoch.pt", type=str,
                        help="obj segmentation model path")
parser.add_argument("--record-to", help="Write received media to a file."),
parser.add_argument("--verbose", "-v", action="count")
parser.add_argument("--save_each", default=None, help="save each inference result during the demo")
parser.add_argument("--dataroot", default="./tmp")
parser.add_argument("--trained_model_prefix", type=str, default="src/trainer/logs/20220214_joint")
parser.add_argument("--joint", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    if args.save_each:
        save_each = join(args.save_each, get_name_by_date())
    else:
        save_each = None
    teacher = Teach(args, args.dataroot, num_classes=3, device="cuda:0", saveroot=save_each)
    opt_teach = TeachOperator(args, teacher)

    assess = Assess(args, args.dataroot, num_classes=3, device="cuda:0")
    opt_assess = AssessOperator(args, assess)
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/teach", index_teach)
    app.router.add_get("/assess", index_assess)
    app.router.add_post("/post_image", upload_file)
    app.router.add_post("/post_status", update_status)
    app.router.add_post("/offer", get_offer)
    app.router.add_static("/assets/", join(STATIC_PATH, "assets/"))

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
