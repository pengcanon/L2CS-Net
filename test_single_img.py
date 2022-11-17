import os, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import glob
from PIL import Image, ImageOps

import datasets
from utils import select_device, draw_gaze
from model import L2CS

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    parser.add_argument(
        '--impath',dest='impath', help='Path of single images.',
        default='singleimages/', type=str
    )
    args = parser.parse_args()
    return args


def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch  == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    arch=args.arch
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    model=getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()


    softmax = nn.Softmax(dim=1)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0
    file_imgs = glob.glob(args.impath + '*.jpg')
    id_img = 0
    with torch.no_grad():
        while True:
            img = cv2.cvtColor(cv2.imread(file_imgs[id_img]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            im_pil = Image.fromarray(img)
            im_pil = transformations(im_pil)
            im_pil = Variable(im_pil).cuda(gpu)
            im_pil = im_pil.unsqueeze(0)

            # gaze prediction
            gaze_pitch, gaze_yaw = model(im_pil)

            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)

            # Get continuous predictions in degrees.
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
            cv2.putText(img, f'angles: {pitch_predicted:.0f}, {yaw_predicted:.0f}', (0, 20),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

            draw_gaze(0,0,224, 224,img,(pitch_predicted,yaw_predicted),color=(0,0,255))


            cv2.imshow("Demo", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(5)
            if key & 0xFF == 27:
                break

            if key & 0xFF == ord('n'):
                id_img += 1
                if id_img > len(file_imgs)-1:
                    id_img = 0

            if key & 0xFF == ord('p'):
                id_img -= 1
                if id_img < 0:
                    id_img = len(file_imgs) - 1