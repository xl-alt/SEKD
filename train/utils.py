from visdom import Visdom
import torch
import torch.nn as nn
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

def netParams(model, input_size=(1,3,512,512)):
    """
    Computing total network parameters and FLOPs
    args:
       model: PyTorch model
       input_size: tuple, the size of the input tensor (e.g., (1, 3, 224, 224) for a single 224x224 RGB image)
    return: the number of parameters and FLOPs
    """
    total_parameters = 0
    total_flops = 0

    # Calculate total parameters
    for parameter in model.parameters():
        param_size = parameter.size()
        total_parameters += param_size.numel()  # Total number of elements in the parameter

    # Calculate FLOPs
    def count_flops(layer, x, y):
        nonlocal total_flops
        if isinstance(layer, torch.nn.Conv2d):
            # FLOPs for Conv2d = 2 * output_channels * output_height * output_width * kernel_height * kernel_width * input_channels / groups
            output_height, output_width = y.size(2), y.size(3)
            kernel_height, kernel_width = layer.kernel_size
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            groups = layer.groups
            flops = 2 * out_channels * output_height * output_width * kernel_height * kernel_width * in_channels / groups
            total_flops += flops

    # Register hooks to calculate FLOPs
    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(count_flops))

    # Forward pass with a dummy input to trigger hooks
    dummy_input = torch.rand(input_size)
    model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_parameters, total_flops

