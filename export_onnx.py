#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import torch
from models.Simplify_Net import Simplify_Net
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Export Onnx', add_help=False)
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--weights', default='./output_dir/best.pth', type=str,help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,help='number of the classification types')

    return parser

def main(args):

    x = torch.randn(1, 3, args.input_size[0],args.input_size[1])
    input_names = ["input"]
    out_names = ["output"]

    model = Simplify_Net(args.nb_classes)

    checkpoint = torch.load(args.weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)

    model.eval()

    torch.onnx.export(model, x, args.weights.replace('pth','onnx'), export_params=True, training=False, input_names=input_names, output_names=out_names)
    print('please run: python -m onnxsim test.onnx test_sim.onnx\n')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
