# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from PIL import Image
from argparse import ArgumentParser


# 添加 Colorize 类的定义
class Colorize:
    def __init__(self, n=3):
        # 定义一个自定义颜色映射，适用于8个类别
        self.cmap = np.array([
            [0, 0, 0],  # 类别 0
            [0, 255, 0],  # 类别 1
            [255, 0, 0],  # 类别 2
            # [255,0,0], # 类别 3
            # [255,0,127], # 类别 4
            # [0,255,0], # 类别 5
            # [255,255,255],  # 类别 6
            # [0,255,255]    # 类别 7
        ], dtype=np.uint8)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


input_transform_cityscapes = Compose([
    Resize((512, 512), Image.BILINEAR),
    ToTensor(),
])


def main(args):
    savedir = f'./预测结果/{args.distillation_type}/SUGGER/CT'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if args.distillation_type == 'student':
        from models.Segformer4EmbeddingKD import mit_b0
        model = mit_b0()
        from knowledge_distillation import FEF
        model = FEF(model, 5)
        weightspath = r'./model.pth'

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print("Model and weights LOADED successfully")
    model = torch.nn.DataParallel(model)

    if not args.cpu:
        model = model.cuda()
    model.eval()

    colorizer = Colorize()  # 创建 Colorize 实例

    for filename in os.listdir(args.input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, filename)
            image = Image.open(image_path).convert('RGB')

            input_tensor = input_transform_cityscapes(image).unsqueeze(0)

            if not args.cpu:
                input_tensor = input_tensor.cuda()

            with torch.no_grad():
                # output = model(input_tensor)
                _, output, _ = model(input_tensor)

            pred = output[0].max(0)[1].byte().cpu().data
            pred_color = colorizer(pred.unsqueeze(0))  # 使用 colorizer 实例

            save_path = os.path.join(savedir, f'pred_{filename}')
            pred_image = ToPILImage()(pred_color)
            pred_image.save(save_path)

            print(f"已处理并保存: {save_path}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--distillation-type', default='student',
                        choices=['teacher', 'student', 'TransKDBase', 'TransKD_GL', 'TransKD_EA'])
    parser.add_argument('--input_dir', default=r'.\predictimg', help='输入图片文件夹路径')
    parser.add_argument('--cpu', default=False,  action='store_true', help='是否使用CPU进行计算')

    main(parser.parse_args())