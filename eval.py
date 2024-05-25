#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import argparse
from utils.transform import Resize,Compose,ToTensor,Normalize
from utils.datasets import SegData
import torch
import os
import numpy as np
from tqdm import tqdm
from models.Simplify_Net import Simplify_Net
from utils.metrics import Evaluator

def get_args_parser():
    parser = argparse.ArgumentParser('Eval Model', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,help='Batch size for training')
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--data_path', default='./datasets/', type=str,help='dataset path')
    parser.add_argument('--weights', default='./output_dir/best.pth', type=str,help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,help='number of the classification types')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def main(args):

    device = torch.device(args.device)

    segmetric = Evaluator(args.nb_classes)
    segmetric.reset()

    val_transform = Compose([
                                    Resize(args.input_size),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    val_dataset = SegData(image_path=os.path.join(args.data_path, 'images/val'),
                            mask_path=os.path.join(args.data_path, 'labels/val'),
                            data_transforms=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    model = Simplify_Net(args.nb_classes)

    checkpoint = torch.load(args.weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)

    model.to(device)
    model.eval()

    classes = ["background","human"]

    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for image, label in val_loader:
                output = model(image.to(device))
                pred = output.data.cpu().numpy()
                label = label.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                segmetric.add_batch(label, pred)
                pbar.update(1)

    pix_acc = segmetric.Pixel_Accuracy()
    every_iou,miou = segmetric.Mean_Intersection_over_Union()

    print("Pixel Accuracy is :", pix_acc)
    print("==========Every IOU==========")
    for name,prob in zip(classes,every_iou):
        print(name+" : "+str(prob))
    print("=============================")
    print("MiOU is :", miou)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
