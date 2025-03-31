import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
# from iouEval import iouEval, getColorEntry

# 假设你有以下已实现的组件：
# 1. 已训练好的模型类，如 mit_b0
# 2. 已定义好的数据集类 CustomVOCDataset
# 3. iouEval类可用来计算 IoU (其中有 confMatrix 属性存储混淆矩阵)
from models.Segformer import mit_b0, mit_b3, mit_b1, mit_x1  # 根据你的实际模型修改
from transform import Relabel, ToLabel, Colorize  # 根据你的实际transform修改
from knowledge_distillation import FAKDLoss,FeatureAlignmentLoss, boundary_loss, MultiScaleContextAlignmentDistillationLoss,SelfSupervisedDistillationLoss, ContrastiveLoss, TextureDistillationLoss # 新的损失计算方法
NUM_CLASSES = 3  # 根据你的数据集类别数修改
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class iouEval:
    def __init__(self, nClasses, ignoreIndex=None):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex
        # 初始化一个 nClasses x nClasses 的0矩阵作为混淆矩阵
        self.confMatrix = np.zeros((nClasses, nClasses), dtype=np.int64)
        self.reset()

    def reset(self):
        self.confMatrix.fill(0)

    def addBatch(self, preds, targets):
        # preds: [B, 1, H, W]
        # targets: [B, 1, H, W]
        # 假设 preds 和 targets 的shape和类型已满足要求
        preds = preds.cpu().numpy().astype(int)
        targets = targets.cpu().numpy().astype(int)
        for p, t in zip(preds, targets):
            # p和t是单张图片的预测和标注
            p = p.flatten()
            t = t.flatten()
            k = (t >= 0) & (t < self.nClasses)
            hist = np.bincount(
                self.nClasses * p[k] + t[k],
                minlength=self.nClasses ** 2
            ).reshape(self.nClasses, self.nClasses)
            self.confMatrix += hist

    def getIoU(self):
        # 根据confMatrix计算IoU
        # IoU = diag(confMatrix) / (sum(confMatrix,1)+sum(confMatrix,0)-diag(confMatrix))
        sum_over_row = self.confMatrix.sum(axis=1)
        sum_over_col = self.confMatrix.sum(axis=0)
        diag = np.diag(self.confMatrix)
        denominator = sum_over_row + sum_over_col - diag
        # 避免除零
        denominator = np.where(denominator == 0, 1, denominator)
        iou = diag / denominator
        mIoU = np.nanmean(iou)
        return mIoU, iou

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * pred[k].astype(int) + label[k].astype(int),
                       minlength=n ** 2).reshape(n, n)

def compute_metrics(iouEvalObj):
    # 从iouEval对象获取IoU和混淆矩阵
    miou, per_class_iou = iouEvalObj.getIoU()
    # 直接访问confMatrix属性
    hist = iouEvalObj.confMatrix.astype(np.float64)

    # Accuracy
    correct_pixels = np.diag(hist).sum()
    total_pixels = hist.sum()
    accuracy = correct_pixels / (total_pixels + 1e-10)

    # Frequency Weighted IoU (F-mIoU)
    freq = hist.sum(axis=1) / (hist.sum() + 1e-10)
    fmiou = (freq * per_class_iou).sum()

    return miou, accuracy, fmiou, per_class_iou

class CustomVOCDataset():
    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform

        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        # 读取图像文件名列表
        self.images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.masks = [os.path.join(mask_dir, f.replace('.jpg', '.png')) for f in os.listdir(image_dir) if
                      f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

class ValTransform(object):
    def __init__(self, height=512, width=512):
        self.height = height
        self.width = width

    def __call__(self, image, label):
        image = image.resize((self.width, self.height), Image.BILINEAR)
        label = label.resize((self.width // 4, self.height // 4), Image.NEAREST)
        image = ToTensor()(image)
        label = ToLabel()(label)
        label = Relabel(255, NUM_CLASSES - 1)(label)
        return image, label

def main():
    # ============= 用户需要修改的地方 =============

    # 已训练好的模型路径
    trained_weights_path = r".\model.pth"

    # 验证集数据集路径
    val_root = r".\VOC2007"

    transform_val = ValTransform(height=512, width=512)  # 根据实际情况修改

    # 创建验证集
    val_dataset = CustomVOCDataset(root=val_root, image_set='val', transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 定义模型
    model = mit_b0(num_classes=NUM_CLASSES, image_size=(512, 512))  # 根据你的模型和输入大小修改
    # model = build_kd_trans1(model,5)
    from ptflops import get_model_complexity_info

    # 3通道图像，高和宽为512
    input_res = (3, 512, 512)  # (C,H,W)

    macs, params = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
    print("the number of parameters: %d ==> %.2f M" % (params, (params / 1e6)))
    print("Computational complexity (FLOPs): %.2f GMac" % (macs / 1e9))
    # 加载已训练权重
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    # 若权重中有 "module." 前缀，则去掉后匹配
                    stripped_name = name.split("module.")[-1]
                    if stripped_name in own_state:
                        own_state[stripped_name].copy_(param)
                else:
                    print(name, " not loaded")
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(trained_weights_path, map_location='cpu'))
    print("Model and weights LOADED successfully")
    model = model.to(DEVICE)
    model.eval()

    # ============= 开始计算mIoU =============
    iouEvalVal = iouEval(NUM_CLASSES)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # _, outputs, _ = model(images)
            outputs = model(images)
            # print(type(outputs))
            # print(outputs)

            preds = outputs.max(1)[1].unsqueeze(1)  # [B,1,H,W]

            iouEvalVal.addBatch(preds.data, labels.data)

    miou, accuracy, fmiou, per_class_iou = compute_metrics(iouEvalVal)

    print("Validation Results:")
    print("Overall Accuracy: {:.2f}%".format(accuracy * 100))
    print("Mean IoU: {:.2f}%".format(miou * 100))
    print("Frequency Weighted IoU: {:.2f}%".format(fmiou * 100))

    # 如果需要打印每类IoU，可添加以下代码
    print("\nPer-Class IoU:")
    for idx, val in enumerate(per_class_iou):
        print(f"Class {idx}: {val*100:.2f}%")

if __name__ == '__main__':
    main()
