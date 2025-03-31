import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
class FeatureEmbedding(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(FeatureEmbedding, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.cbam1 = CBAM(mid_channel)  # Apply CBAM after the first convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.cbam2 = CBAM(out_channel)  # Apply CBAM after the second convolution

    def forward(self, x):
        x = self.conv1(x)
        x = self.cbam1(x)  # Apply CBAM
        x = self.conv2(x)
        x = self.cbam2(x)  # Apply CBAM
        return x

class FEA(nn.Module):
    def __init__(self, student, in_channels, out_channels, mid_channel, embed):
        super(FEA, self).__init__()
        self.student = student
        self.embed = embed
        self.embeddings = nn.ModuleList()

        for in_ch, out_ch in zip(in_channels, out_channels):
            self.embeddings.append(FeatureEmbedding(in_ch, mid_channel, out_ch))

        if self.embed > 0:
            self.embed_projections = nn.ModuleList()
            for i in range(len(in_channels)):
                self.embed_projections.append(nn.Linear(in_channels[i], out_channels[i]))

    def forward(self, x):
        student_features = self.student(x)
        embed_features = student_features[2]
        logit = student_features[1]
        features = student_features[0]

        results = []
        for i, feature in enumerate(features):
            embedded = self.embeddings[i](feature)
            results.append(embedded)

        if self.embed == 0:
            return results, logit
        else:
            embedproj = []
            if self.embed == 5:
                for i in range(len(embed_features)):
                    embedproj.append(self.embed_projections[i](embed_features[i]))
            else:
                embedproj.append(self.embed_projections[self.embed - 1](embed_features[self.embed - 1]))
            return results, logit, embedproj

def FEF(model, embed, in_channels=[32, 64, 160, 256], out_channels=[64, 128, 320, 512]):
    # print(embed)
    mid_channel = 64
    student = model
    model = FEA(student, in_channels, out_channels, mid_channel, embed)
    return model






# 测试代码
# if __name__ == "__main__":
#     # 创建学生模型
#     student_model = SimpleStudentModel()
#
#     # 构建FPN模型
#     fpn_model = build_fpn_fusion(
#         model=student_model,
#         embed=5,
#         in_channels=[16, 32, 64, 128],  # 与学生模型的特征通道数匹配
#         out_channels=[64, 128, 256, 512]  # FPN输出通道数
#     )
#
#     # 创建示例输入
#     input_tensor = torch.randn(1, 3, 224, 224)
#
#     # 运行模型
#     results, logit, embedproj = fpn_model(input_tensor)
#
#     # 打印结果
#     print("FPN output shapes:", [r.shape for r in results])
#     print("Logit shape:", logit.shape)
#     print("Embed projection shapes:", [e.shape for e in embedproj])




class FeatureDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(FeatureDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))



# 自适应权重部分
class AdaptiveWeights(nn.Module):
    def __init__(self, num_features):
        super(AdaptiveWeights, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_features))

    def forward(self):
        return F.softmax(self.weights, dim=0)


class AdaptiveHCL(nn.Module):
    def __init__(self, num_features):
        super(AdaptiveHCL, self).__init__()
        self.adaptive_weights = AdaptiveWeights(num_features)

    def forward(self, fstudent, fteacher):
        assert len(fstudent) == len(fteacher), "学生和教师的特征数量必须相同"

        weights = self.adaptive_weights()
        loss_all = 0.0

        for fs, ft, weight in zip(fstudent, fteacher, weights):
            n, c, h, w = fs.shape
            loss = F.mse_loss(fs, ft, reduction='mean')

            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                tmpft = F.adaptive_avg_pool2d(ft, (l, l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt

            loss = loss / tot
            loss_all += weight * loss

        return loss_all

class MultiScaleContextAlignmentDistillationLoss(nn.Module):
    def __init__(self, scales=[1, 1, 1], alpha=1, beta=1):
        super(MultiScaleContextAlignmentDistillationLoss, self).__init__()
        self.scales = scales
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, student_features, teacher_features):

        total_loss = 0.0
        for scale in self.scales:
            for s_feat, t_feat in zip(student_features, teacher_features):
                # 调整特征图的尺度
                if scale != 1:
                    s_feat_scaled = F.interpolate(s_feat, scale_factor=scale, mode='bilinear', align_corners=False)
                    t_feat_scaled = F.interpolate(t_feat, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    s_feat_scaled = s_feat
                    t_feat_scaled = t_feat

                # 特征对齐损失
                feature_loss = self.mse_loss(s_feat_scaled, t_feat_scaled) # Mes

                # 上下文关系对齐损失
                # 计算关系矩阵 (batch, C, H*W)
                s_rel = torch.bmm(s_feat_scaled.view(s_feat_scaled.size(0), s_feat_scaled.size(1), -1),
                                  s_feat_scaled.view(s_feat_scaled.size(0), s_feat_scaled.size(1), -1).transpose(1, 2))
                t_rel = torch.bmm(t_feat_scaled.view(t_feat_scaled.size(0), t_feat_scaled.size(1), -1),
                                  t_feat_scaled.view(t_feat_scaled.size(0), t_feat_scaled.size(1), -1).transpose(1, 2))
                # 归一化关系矩阵
                s_rel = F.normalize(s_rel, p=2, dim=1)
                t_rel = F.normalize(t_rel, p=2, dim=1)
                context_loss = self.mse_loss(s_rel, t_rel)

                # 总损失
                loss =  feature_loss + context_loss
                total_loss += loss

        # 平均多尺度损失
        total_loss = total_loss / (len(self.scales) * len(student_features))
        return total_loss


class FeatureAugmentation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAugmentation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAugmentation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAugmentation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FAKDLoss(nn.Module):
    def __init__(self, scales=[1]):
        """
        :param scales: 特征对齐的缩放因子列表。例如，[1.0, 0.5, 0.25] 表示不同的尺度。
        """
        super(FAKDLoss, self).__init__()
        self.scales = scales
        self.mse_loss = nn.MSELoss()
        self.augmentation_modules = None
        self.channel_align = None

    def _initialize_modules(self, student_features, teacher_features):
        student_channels = [feat.shape[1] for feat in student_features]
        teacher_channels = [feat.shape[1] for feat in teacher_features]
        device = student_features[0].device  # 获取输入特征的设备

        self.augmentation_modules = nn.ModuleList([
            FeatureAugmentation(s_c, s_c).to(device) for s_c in student_channels
        ])

        self.channel_align = nn.ModuleList([
            nn.Conv2d(s_c, t_c, kernel_size=1, bias=False).to(device)
            for s_c, t_c in zip(student_channels, teacher_channels)
        ])

    def augment_features(self, features):
        return [aug(feat) for aug, feat in zip(self.augmentation_modules, features)]

    def forward(self, student_features, teacher_features):
        assert len(student_features) == len(teacher_features), "学生和教师特征列表长度必须相同"

        if self.augmentation_modules is None or self.channel_align is None:
            self._initialize_modules(student_features, teacher_features)

        total_loss = 0.0
        num_alignments = 0
        augmented_teacher_features = self.augment_features(teacher_features)

        for idx, (s_feat, t_feat) in enumerate(zip(student_features, augmented_teacher_features)):
            for scale in self.scales:
                # 尺度对齐
                if scale != 1.0:
                    s_feat_scaled = F.interpolate(s_feat, scale_factor=scale, mode='bilinear', align_corners=False)
                    t_feat_scaled = F.interpolate(t_feat, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    s_feat_scaled = s_feat
                    t_feat_scaled = t_feat

                # 通道对齐
                s_feat_aligned = self.channel_align[idx](s_feat_scaled)

                # 确保教师特征的通道数与学生对齐后的特征相同
                if s_feat_aligned.shape[1] != t_feat_scaled.shape[1]:
                    raise ValueError(
                        f"教师特征的通道数 {t_feat_scaled.shape[1]} 与学生对齐后的通道数 {s_feat_aligned.shape[1]} 不匹配")

                # 特征对齐损失
                loss = self.mse_loss(s_feat_aligned, t_feat_scaled)
                total_loss += loss
                num_alignments += 1

        # 计算平均损失
        if num_alignments > 0:
            total_loss = total_loss / num_alignments
        return total_loss

    def to(self, device):
        """
        将模块移动到指定设备
        """
        super(FAKDLoss, self).to(device)
        if self.augmentation_modules is not None:
            self.augmentation_modules = self.augmentation_modules.to(device)
        if self.channel_align is not None:
            self.channel_align = self.channel_align.to(device)
        return self




class FeatureAlignmentLoss(nn.Module):
    def __init__(self, scales=[0.25], student_channels=[], teacher_channels=[]):
        """
        :param scales: 特征对齐的缩放因子列表。例如，[1.0, 0.5, 0.25] 表示不同的尺度。
        :param student_channels: 学生模型各层特征的通道数列表。
        :param teacher_channels: 教师模型各层特征的通道数列表。
        """
        super(FeatureAlignmentLoss, self).__init__()
        self.scales = scales
        self.mse_loss = nn.MSELoss()

        # 如果学生和教师的通道数不同，使用1x1卷积调整学生特征的通道数
        if student_channels and teacher_channels:
            assert len(student_channels) == len(teacher_channels), "学生和教师通道数列表长度必须相同"
            self.channel_align = nn.ModuleList([
                nn.Conv2d(s_c, t_c, kernel_size=1, bias=False)
                for s_c, t_c in zip(student_channels, teacher_channels)
            ])
        else:
            self.channel_align = None

    def forward(self, student_features, teacher_features):
        """
        :param student_features: 学生模型的中间特征列表 [feat1, feat2, ...]
        :param teacher_features: 教师模型的中间特征列表 [feat1, feat2, ...]
        :return: 总特征对齐损失
        """
        assert len(student_features) == len(teacher_features), "学生和教师特征列表长度必须相同"

        total_loss = 0.0
        num_alignments = 0

        for idx, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            for scale in self.scales:
                # 尺度对齐
                if scale != 1.0:
                    s_feat_scaled = F.interpolate(s_feat, scale_factor=scale, mode='bilinear', align_corners=False)
                    t_feat_scaled = F.interpolate(t_feat, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    s_feat_scaled = s_feat
                    t_feat_scaled = t_feat

                # 通道对齐
                if self.channel_align:
                    s_feat_aligned = self.channel_align[idx](s_feat_scaled)
                else:
                    s_feat_aligned = s_feat_scaled

                # 确保教师特征的通道数与学生对齐后的特征相同
                if s_feat_aligned.shape[1] != t_feat_scaled.shape[1]:
                    raise ValueError(
                        f"教师特征的通道数 {t_feat_scaled.shape[1]} 与学生对齐后的通道数 {s_feat_aligned.shape[1]} 不匹配")

                # 特征对齐损失
                loss = self.mse_loss(s_feat_aligned, t_feat_scaled)
                total_loss += loss
                num_alignments += 1

        # 计算平均损失
        if num_alignments > 0:
            total_loss = total_loss / num_alignments
        return total_loss


# 边界函数
def extract_boundary(embedding, method='sobel'):
    """提取embedding中的边界信息"""
    device = embedding.device  # 获取embedding所在的设备
    b, hw, c = embedding.shape
    h = w = int(hw ** 0.5)
    embedding = embedding.view(b, h, w, c).permute(0, 3, 1, 2)

    if method == 'sobel':
        kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = torch.tensor([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    elif method == 'laplacian':
        kernel_x = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = kernel_x  # Laplacian是各向同性的

    padding = (1, 1, 1, 1)
    # 防止卷积结果过大导致溢出
    edge_x = F.conv2d(F.pad(embedding, padding), kernel_x.repeat(c, 1, 1, 1), groups=c)
    edge_y = F.conv2d(F.pad(embedding, padding), kernel_y.repeat(c, 1, 1, 1), groups=c)

    # Clamp防止数值过大
    edge_x = torch.clamp(edge_x, min=-1e6, max=1e6)
    edge_y = torch.clamp(edge_y, min=-1e6, max=1e6)

    # 添加一个小常数以防止数值不稳定
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

    return edge.view(b, c, -1).permute(0, 2, 1)


def boundary_loss(teacher_embeds, student_embeds, method='sobel', alpha=0.5):
    total_loss = 0
    for t_emb, s_emb in zip(teacher_embeds, student_embeds):
        t_boundary = extract_boundary(t_emb, method)
        s_boundary = extract_boundary(s_emb, method)

        # 检查是否有NaN或Inf
        if not torch.isfinite(t_boundary).all() or not torch.isfinite(s_boundary).all():
            continue  # 跳过这一对嵌入

        # 使用L1损失计算边界差异
        l1_loss = F.l1_loss(t_boundary, s_boundary)

        # 计算额外的损失，例如平滑损失
        smooth_loss = F.mse_loss(t_boundary, F.avg_pool2d(t_boundary, kernel_size=3, stride=1, padding=1))

        # 检查损失是否为NaN
        if torch.isnan(l1_loss) or torch.isnan(smooth_loss):
            continue  # 跳过这一对嵌入

        # 总损失为L1损失和额外损失的加权和
        loss = l1_loss + alpha * smooth_loss
        total_loss += loss

    # 防止除以零
    if len(teacher_embeds) == 0:
        return torch.tensor(0.0, device=teacher_embeds[0].device)

    return total_loss / len(teacher_embeds)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, student_features_list, teacher_features_list):
        if len(student_features_list) != len(teacher_features_list):
            raise ValueError("学生和教师的特征列表长度必须相同")

        total_loss = 0.0
        for student_feat, teacher_feat in zip(student_features_list, teacher_features_list):
            # 如果特征为4维 B x C x H x W，先进行全局平均池化转化为 B x C
            if student_feat.dim() == 4:
                student_feat = student_feat.mean(dim=[2, 3])  # 全局平均池化
            if teacher_feat.dim() == 4:
                teacher_feat = teacher_feat.mean(dim=[2, 3])

            # 确保特征是 B x C 二维张量
            if student_feat.dim() != 2 or teacher_feat.dim() != 2:
                raise ValueError("特征必须是二维张量 (B x C)")

            # 对特征归一化
            student_norm = F.normalize(student_feat, dim=1)
            teacher_norm = F.normalize(teacher_feat, dim=1)

            # 计算相似度矩阵
            similarity_matrix = torch.matmul(student_norm, teacher_norm.mT) / self.temperature

            # 生成标签（假设对应位置为相同类别）
            labels = torch.arange(student_feat.size(0)).to(student_feat.device)

            # 使用交叉熵计算对比损失
            loss = F.cross_entropy(similarity_matrix, labels)
            total_loss += loss

        # 根据 reduction 参数进行最终的聚合
        if self.reduction == 'mean':
            total_loss = total_loss / len(student_features_list)
        elif self.reduction == 'sum':
            pass
        else:
            raise ValueError("reduction 参数只能是 'mean' 或 'sum'")

        return total_loss

import math
class TextureDistillationLoss(nn.Module):
    def __init__(self, img_size=128, patch_size=1, num_levels=50, alpha=0.3, theta=0.9):
        """
        初始化纹理蒸馏损失
        Args:
            img_size (int): 特征图的边长
            patch_size (int): patch的大小
            num_levels (int): 量化级别数
            alpha (float): 量化比例系数
            theta (float): intensity限制阈值
        """
        super(TextureDistillationLoss, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_levels = num_levels
        self.alpha = alpha
        self.theta = theta
        self.gaussian_kernel = None

    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0, device='cuda'):
        """
        创建高斯核并确保在正确的设备上
        Args:
            kernel_size: 卷积核大小
            sigma: 高斯分布的标准差
            device: 设备类型
        """
        if self.gaussian_kernel is None:
            coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
            coords = coords.expand(kernel_size, kernel_size)
            gaussian = torch.exp(-(coords ** 2 + coords.t() ** 2) / (2 * sigma ** 2))
            gaussian = gaussian / gaussian.sum()
            self.gaussian_kernel = gaussian.unsqueeze(0).unsqueeze(0)
        return self.gaussian_kernel

    def _reshape_to_image(self, x):
        """
        将transformer格式的embeds重塑为图像格式
        Args:
            x: [B, N, C] 格式的embeds
        Returns:
            [B, C, H, W] 格式的特征图
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))  # N = H * W
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def _compute_structural_texture(self, x):
        """
        计算结构纹理特征
        Args:
            x: [B, C, H, W] 格式的特征图
        Returns:
            list of tensors: 拉普拉斯金字塔分解结果
        """
        device = x.device
        gaussian_kernel = self._create_gaussian_kernel(device=device).to(device)

        current = x
        pyramids = []

        for _ in range(2):  # 2层分解
            expanded_kernel = gaussian_kernel.expand(current.size(1), -1, -1, -1).to(device)

            # 应用高斯模糊
            blurred = F.conv2d(
                current,
                expanded_kernel,
                padding=2,
                groups=current.size(1)
            )

            # 计算拉普拉斯差值
            laplace = current - blurred
            pyramids.append(laplace)

            # 下采样作为下一层输入
            current = F.avg_pool2d(blurred, 2)

        pyramids.append(current)
        return pyramids

    def _compute_statistical_texture(self, x):
        """
        计算统计纹理特征
        Args:
            x: [B, C, H, W] 格式的特征图
        Returns:
            tensor: 归一化的统计特征
        """
        B, C, H, W = x.shape

        # 使用unfold操作获取局部区域
        patch_size = 4  # 可调整的局部区域大小
        patches = F.unfold(x,
                           kernel_size=patch_size,
                           stride=patch_size // 2,
                           padding=patch_size // 2)

        # 计算每个patch的统计量
        patches = patches.reshape(B, C, patch_size * patch_size, -1)
        mean = patches.mean(dim=2, keepdim=True)
        std = patches.std(dim=2, keepdim=True)

        # 计算归一化的统计特征
        normalized = (patches - mean) / (std + 1e-6)

        return normalized

    def _compute_texture_loss(self, teacher_feat, student_feat):
        """
        计算纹理损失
        Args:
            teacher_feat: 教师网络特征
            student_feat: 学生网络特征
        Returns:
            tuple: (结构损失, 统计损失)
        """
        # 结构纹理损失
        t_struct = self._compute_structural_texture(teacher_feat)
        s_struct = self._compute_structural_texture(student_feat)

        struct_loss = sum(F.mse_loss(t, s) for t, s in zip(t_struct, s_struct))

        # 统计纹理损失
        t_stat = self._compute_statistical_texture(teacher_feat)
        s_stat = self._compute_statistical_texture(student_feat)

        stat_loss = F.mse_loss(t_stat, s_stat)

        return struct_loss, stat_loss

    def forward(self, teacher_embeds, student_embeds):
        """
        计算总的纹理蒸馏损失
        Args:
            teacher_embeds: list of teacher network embeds [B, N, C]
            student_embeds: list of student network embeds [B, N, C]
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # 确保输入是列表
        if not isinstance(teacher_embeds, list):
            teacher_embeds = [teacher_embeds]
        if not isinstance(student_embeds, list):
            student_embeds = [student_embeds]

        total_struct_loss = 0
        total_stat_loss = 0

        # 确保teacher和student的特征数量相同
        assert len(teacher_embeds) == len(student_embeds), \
            f"Teacher ({len(teacher_embeds)}) and student ({len(student_embeds)}) should have same number of features"

        # 获取设备信息
        device = teacher_embeds[0].device

        for t_embed, s_embed in zip(teacher_embeds, student_embeds):
            # 确保所有张量都在同一个设备上
            t_embed = t_embed.to(device)
            s_embed = s_embed.to(device)

            # 转换为图像格式
            t_feat = self._reshape_to_image(t_embed)
            s_feat = self._reshape_to_image(s_embed)

            # 计算纹理损失
            struct_loss, stat_loss = self._compute_texture_loss(t_feat, s_feat)

            total_struct_loss += struct_loss
            total_stat_loss += stat_loss

        # 计算平均损失
        num_levels = len(teacher_embeds)
        total_struct_loss /= num_levels
        total_stat_loss /= num_levels

        # 可以调整这两种损失的权重
        struct_weight = 1.0
        stat_weight = 1.0
        total_loss = struct_weight * total_struct_loss + stat_weight * total_stat_loss

        loss_dict = {
            'total_loss': total_loss,
            'structural_loss': total_struct_loss,
            'statistical_loss': total_stat_loss
        }

        return total_loss


class SelfSupervisedDistillationLoss(torch.nn.Module):
    def __init__(self):
        super(SelfSupervisedDistillationLoss, self).__init__()
    def feature_reconstruction_loss(self, student_feat, teacher_feat):
        """特征重建任务的损失
        Args:
            student_feat: 学生模型特征 [B, C, H, W]
            teacher_feat: 教师模型特征 [B, C, H, W]
        """
        # 随机mask一些区域
        B, C, H, W = student_feat.size()
        mask = torch.ones_like(student_feat)
        # 随机选择要mask的区域
        mask_ratio = 0.3
        mask_h = int(H * mask_ratio)
        mask_w = int(W * mask_ratio)
        h_start = torch.randint(0, H - mask_h, (B,))
        w_start = torch.randint(0, W - mask_w, (B,))

        for b in range(B):
            mask[b, :, h_start[b]:h_start[b] + mask_h,
            w_start[b]:w_start[b] + mask_w] = 0

        # 计算重建损失
        masked_student = student_feat * mask
        reconstruction_loss = F.mse_loss(masked_student, teacher_feat)

        return reconstruction_loss

    def forward(self, student_features, teacher_features):
        """计算自监督蒸馏损失
        Args:
            student_features: 学生模型特征列表
            teacher_features: 教师模型特征列表
        """
        rotation_loss = 0
        reconstruction_loss = 0

        # 对每一层特征计算自监督损失
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 特征重建损失
            reconstruction_loss += self.feature_reconstruction_loss(
                s_feat, t_feat)

        total_loss = reconstruction_loss
        return total_loss / len(student_features)
