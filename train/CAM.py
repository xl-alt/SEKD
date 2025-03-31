# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import os
from models.Segformer4EmbeddingKD import mit_b0


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # 注册hooks
        self.handle_forward = target_layer.register_forward_hook(self.forward_hook)
        self.handle_backward = target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, class_idx=None):
        if self.gradients is None or self.activations is None:
            print("GradCAM: No gradients or activations found!")
            return None
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)  # [1,1,H,W]
        cam = F.relu(cam)  # 只保留正值部分
        cam = cam.detach().cpu().numpy()
        cam = cam[0, 0, ...]  # 转成 (H, W)
        # 归一化处理
        if np.max(cam) == np.min(cam):
            return None
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam

    def remove_hooks(self):
        self.handle_forward.remove()
        self.handle_backward.remove()


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


def load_model(model_path, num_classes=8):
    from models.Segformer4EmbeddingKD import mit_b0
    model = mit_b0()
    from CSF import build_kd_trans1
    model = build_kd_trans1(model, 5)
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

    model = load_my_state_dict(model, torch.load(model_path))
    print("Model and weights LOADED successfully")
    # model = torch.nn.DataParallel(model)

    model = model.cuda()
    model.eval()
    return model


def preprocess_image(image_path, input_size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, original_image


def generate_gradcam_overlay(cam, original_image, alpha=0.5):
    if cam is None:
        return None
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)
    return overlay


if __name__ == "__main__":
    # 模型权重和输入/输出路径 (请根据实际情况修改)
    model_path = r"E:\开题报告文件夹\论文代码复现\TransKD-main\datalogs\2025newtrain\distilla_b3_b0_空间注意力\dilltila_save_test\Testbatch4-VOC\Baseline\model_SegformerB0-CSF-weight_hlc-nonpretrained-2025-2-17_best.pth"
    input_dir = r"E:\开题报告文件夹\论文代码复现\TransKD-main\datalogs\cam_o\VOCswPaper"
    output_dir = r"E:\开题报告文件夹\论文代码复现\TransKD-main\datalogs\VOCswPaper\融合部分的注意力结果\仅空间1"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_model(model_path)
    print(model)

    # 依据模型结构选择合适的目标层
    # 以下仅为示例，请根据你的模型结构选取对应的层
    target_layer = model.student.patch_embed3.proj
    gradcam = GradCAM(model, target_layer)

    # 列出输入目录中图像文件
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print("Number of image files found:", len(input_files))

    device = next(model.parameters()).device

    for filename in input_files:
        image_path = os.path.join(input_dir, filename)
        print("Processing:", image_path)

        input_tensor, original_image = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)

        _,outputs,_ = model(input_tensor)  # [1, num_classes, H, W]
        print("Outputs shape:", outputs.shape)

        class_scores = outputs.mean(dim=(2, 3))
        class_idx = torch.argmax(class_scores, dim=1).item()
        print("Class index:", class_idx)

        model.zero_grad()
        class_score = outputs[0, class_idx].mean()
        print("Class score:", class_score.item())
        class_score.backward(retain_graph=True)

        print("Gradients is None:", gradcam.gradients is None)
        print("Activations is None:", gradcam.activations is None)

        cam = gradcam.generate(class_idx=class_idx)
        overlay = generate_gradcam_overlay(cam, original_image)

        if overlay is not None:
            # 转换BGR到RGB以适应PIL保存
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(overlay_rgb)

            save_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_gradcam.jpg')
            print("Attempting to save:", save_path)
            pil_image.save(save_path, quality=95)
            print(f"Saved to {save_path}")
        else:
            print("No overlay generated for:", filename)

    gradcam.remove_hooks()
