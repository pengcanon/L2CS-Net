import argparse
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

#from face_detection import RetinaFace
from model import L2CS


batch_size = 1
snapshot_path = 'gaze360.pkl'
model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90)
transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cpu'))
model.load_state_dict(saved_state_dict)
model.eval() #turn on evaluation mode so that layers such as bn and dropout will behave in a normal way
softmax = nn.Softmax(dim=1)
#detector = RetinaFace(gpu_id=0)

idx_tensor = [idx for idx in range(90)]
idx_tensor = torch.FloatTensor(idx_tensor)

img = cv2.imread('a.j.-cook-7.jpg')
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)
img = transformations(im_pil)
#img = Variable(img).cuda(gpu)
img = img.unsqueeze(0)

# gaze prediction
gaze_pitch, gaze_yaw = model(img)

pitch_predicted = softmax(gaze_pitch)
yaw_predicted = softmax(gaze_yaw)
#yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
print(torch.sum(pitch_predicted.data[0] * idx_tensor))
print(pitch_predicted.shape)
print(img.shape)
