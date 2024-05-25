#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import argparse
import torch
import numpy as np
import torchvision.transforms as T
from models.Simplify_Net import Simplify_Net
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('Predict Image', add_help=False)
    parser.add_argument('--image_path', default='./people-man-model-glasses-46219.jpeg', type=str, metavar='MODEL',help='Name of model to train')
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--weights', default='./output_dir/last.pth', type=str,help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,help='number of the classification types')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')

    return parser


def main(args):
    device = torch.device(args.device)

    image = Image.open(args.image_path).convert('RGB')
    img_size = image.size

    transforms = T.Compose([
        T.Resize(args.input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    model = Simplify_Net(args.nb_classes)

    checkpoint = torch.load(args.weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)

    model.to(device)
    model.eval()

    input_tensor = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    mask = Image.fromarray(pred)
    out = mask.resize(img_size)
    out.save("result.png")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
