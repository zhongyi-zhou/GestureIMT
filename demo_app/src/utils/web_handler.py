import json
import io
import base64
from PIL import Image
import numpy as np
import cv2

def img_decoder(mydict, savepath=None):
    base64_decoded = base64.b64decode(mydict["image"].split(",")[1])
    img = np.array(Image.open(io.BytesIO(base64_decoded)))
    # print(img.shape)
    img = img[:,:,:3]
    if savepath:
        cv2.imwrite(savepath, img[:,:,::-1])
    return img