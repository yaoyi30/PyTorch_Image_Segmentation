#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import os
import torch
import torch.nn as nn
from models.Simplify_Net import Simplify_Net
from utils.engine import train_and_val,plot_pix_acc,plot_miou,plot_loss,plot_lr
import argparse
import numpy as np
from utils.transform import Resize,Compose,ToTensor,Normalize,RandomHorizontalFlip
from utils.datasets import SegData

def get_args_parser():
    parser = argparse.ArgumentParser('Image Segmentation Train', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,help='Batch size for training')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--data_path', default='./datasets/', type=str,help='dataset path')

    parser.add_argument('--init_lr', default=1e-5, type=float,help='intial lr')
    parser.add_argument('--max_lr', default=1e-3, type=float,help='max lr')
    parser.add_argument('--weight_decay', default=1e-5, type=float,help='weight decay')

    parser.add_argument('--nb_classes', default=2, type=int,help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser



def main(args):

    device = torch.device(args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_transform = Compose([
                                    Resize(args.input_size),
                                    RandomHorizontalFlip(0.5),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])

    val_transform = Compose([
                                    Resize(args.input_size),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_dataset = SegData(image_path=os.path.join(args.data_path, 'images/train'),
                            mask_path=os.path.join(args.data_path, 'labels/train'),
                            data_transforms=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

    val_dataset = SegData(image_path=os.path.join(args.data_path, 'images/val'),
                            mask_path=os.path.join(args.data_path, 'labels/val'),
                            data_transforms=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    model = Simplify_Net(args.nb_classes)
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, total_steps=args.epochs, verbose=True)

    history = train_and_val(args.epochs, model, train_loader,val_loader,loss_function, optimizer,scheduler,args.output_dir,device,args.nb_classes)

    plot_loss(np.arange(0,args.epochs),args.output_dir, history)
    plot_pix_acc(np.arange(0,args.epochs),args.output_dir, history)
    plot_miou(np.arange(0,args.epochs),args.output_dir, history)
    plot_lr(np.arange(0,args.epochs),args.output_dir, history)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
