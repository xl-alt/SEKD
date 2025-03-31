# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
from pickle import FALSE
import random
import time
import numpy as np
import torch
import math
# import utils #visdom
import datetime
from PIL import Image, ImageOps
from argparse import ArgumentParser
import torch.nn as nn
from torch.optim import SGD, Adam, lr_scheduler, AdamW
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from torch_poly_lr_decay import PolynomialLRDecay
from utils import netParams
from models.Segformer4EmbeddingKD import mit_b0,mit_b1, mit_b2, mit_x1 #,mit_b3,mit_b4,mit_b5, mit_x1,mit_b3
from models.Segformer4EmbeddingKD_teacher import mit_b3
from knowledge_distillation import FeatureAlignmentLoss, MultiScaleContextAlignmentDistillationLoss, FEF, SelfSupervisedDistillationLoss, TextureDistillationLoss # 新的损失计算方法

from dataset import VOC12,cityscapes, ACDC
from transform import Relabel, ToLabel, Colorize
import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile
NUM_CHANNELS = 3
NUM_CLASSES = 3

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

NOW = datetime.datetime.now()
TODAY = f'{NOW.year}-{NOW.month}-{NOW.day}'


class CustomVOCDataset():
    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform

        # 更新为VOC2012标准路径
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'SegmentationClass')

        # 读取分割任务的txt文件
        split_f = os.path.join(root, 'ImageSets', 'Segmentation', image_set + '.txt')
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(self.mask_dir, x + ".png") for x in file_names]

        assert (len(self.images) == len(self.masks))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

class MyCoTransform(object):
    def __init__(self, augment=True, width=512, height=512, model='SegformerB0'):
        # self.enc=enc
        self.augment = augment
        self.height = height
        self.width = width
        self.model = model
        pass

    def __call__(self, input, target):
        input = Resize((self.height, self.width), Image.BILINEAR)(input)

        W, H = input.size

        if self.model.startswith('Segformer'):
            target = Resize((self.height // 4, self.width // 4), Image.NEAREST)(target)
        else:
            target = Resize((self.height, self.width), Image.NEAREST)(target)

        if (self.augment):
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0), fill=255)
            input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
            target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, NUM_CLASSES - 1)(target)
        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)



def train(args, model, teacher):
    best_acc = 0


    weight = torch.ones(NUM_CLASSES)
    weight[0] = 2.5959737  # background
    weight[1] = 6.741505  # person
    weight[2] = 3.5353868  # bird
    # weight[3] = 9.866315  # cat
    # weight[4] = 9.690922  # cow
    # weight[5] = 9.369371  # dog
    # weight[6] = 10.289124  # horse
    # weight[7] = 9.953209  # sheep
    # weight[8] = 4.3098087  # aeroplane
    # weight[9] = 9.490392  # bicycle
    # weight[10] = 7.674411  # boat
    # weight[11] = 9.396925  # bus
    # weight[12] = 10.347794  # car
    # weight[13] = 6.3928986  # motorbike
    # weight[14] = 10.226673  # train
    # weight[15] = 10.241072  # bottle
    # weight[16] = 10.28059  # chair
    # weight[17] = 10.396977  # dining table
    # weight[18] = 10.05567  # potted plant
    # weight[19] = 9.49802  # sofa
    # weight[20] = 9.89054  # tv/monitor

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(augment=True, height=args.height, model=args.model, width=args.width)
    co_transform_val = MyCoTransform(augment=False, height=args.height, model=args.model, width=args.width)
    if args.dataset == 'cityscapes':
        dataset_train = cityscapes(args.datadir, co_transform, 'train')
        dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    elif args.dataset == 'VOC':
        dataset_train = CustomVOCDataset(args.datadir, image_set='train', transform=co_transform)
        dataset_val = CustomVOCDataset(args.datadir, image_set='val', transform=co_transform_val)
    else:
        assert 'Dataset does not exist'

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        weight = weight.to(args.device)
    criterion = CrossEntropyLoss2d().to(args.device)
    if args.knowledge_distillation_loss == 'KL':
        norm_type = 'spatial'
    else:
        norm_type = 'channel'
    # criterion_cwd = CriterionCWD(norm_type,args.divergence,args.temperature).to(args.device)
    print(type(criterion))

    if args.dataset == 'cityscapes':
        savedir = f'../logs/save_mit_x1/Testbatch{args.batch_size}/Baseline'
    elif args.dataset == 'VOC':
        savedir = f'../datalogs/VOCSUGGER/distilla_b3_b0_2025_onlyembeds/dilltila_save_test/Testbatch{args.batch_size}-VOC/Baseline'
    else:
        assert 'Dataset does not exist'

    if args.student_pretrained:
        pretrained = 'pretrained'
    else:
        pretrained = 'nonpretrained'

    if args.savedate:
        savefile = f'{args.model}-{args.knowledge_distillation_feature_fusion}-{args.knowledge_distillation_loss}-{pretrained}-{TODAY}'
    else:    
        savefile = f'{args.model}-{args.knowledge_distillation_feature_fusion}-{args.knowledge_distillation_loss}-{pretrained}'
    automated_log_path = savedir + f"/automated_log_{savefile}.txt"
    modeltxtpath = savedir + f"/model_{savefile}.txt"

    if (not os.path.exists(automated_log_path)):   
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    if args.model.startswith('Segformer'):
        optimizer = AdamW(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=0.01)
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=1500, end_learning_rate=0.0, power=1.0)
    else: 
        assert 'model not supported'
    start_epoch = 1

    if args.visualize and args.steps_plot > 0:
        from visualize import Dashboard 
        board = Dashboard(args.port)
    if args.teacher_val:
        print("----- TEACHER VALIDATING-----")
        teacher.eval()
        epoch_loss_val = []
        time_val = []
        doIouVal =  args.iouVal  
        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        with torch.no_grad():
            for step, (images, labels) in enumerate(loader_val):
                start_time = time.time()
                if args.cuda:
                    images = images.to(args.device)
                    labels = labels.to(args.device)

                inputs = Variable(images)   
                targets = Variable(labels)
                _,outputs,_ = teacher(inputs)
                loss = criterion(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())
                time_val.append(time.time() - start_time)


                if (doIouVal):
                    iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
                    print(f'VAL loss: {average:0.4} ( step: {step})', 
                                "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

            iouVal = 0
            if (doIouVal):
                iouVal, iou_classes = iouEvalVal.getIoU()
                iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
                print ("EPOCH IoU on VAL set: ", iouStr, "%") 

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")


        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        teacher.eval()
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()

            inputs = Variable(images).to(args.device)
            targets = Variable(labels).to(args.device)
            print(inputs.shape,targets.shape)
            if args.embed:
                fstudent,outputs,estudent = model(inputs)
                fteacher,_,eteacher = teacher(inputs)
            else:
                fstudent,outputs = model(inputs)
                fteacher,_,_ = teacher(inputs)
            loss = 0.0
            loss_logits = criterion(outputs, targets[:, 0])
            loss += loss_logits
            optimizer.zero_grad()
            if args.knowledge_distillation_loss == 'MultiScaleContextAlignmentDistillationLoss':

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                adaptive_hcl = MultiScaleContextAlignmentDistillationLoss().to(device)
                loss_our = adaptive_hcl(fstudent, fteacher)
                #print("loss_our:", loss_our)
                loss += 0.5 * loss_our
            elif args.knowledge_distillation_loss == 'SelfSupervisedDistillationLoss':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                adaptive_hcl1 = SelfSupervisedDistillationLoss().to(device)
                loss_our1 = adaptive_hcl1(fstudent, fteacher)
                loss += 0.5 * loss_our1
            elif args.knowledge_distillation_loss == 'FeatureAlignmentLoss':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                adaptive_hcl2 = FeatureAlignmentLoss().to(device)
                loss_our2 = adaptive_hcl2(fstudent, fteacher)
                loss += 0.2 * loss_our2
            if args.embed!=0:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                texture = TextureDistillationLoss()
                lossBound = texture(estudent, eteacher).to(device)
                loss += lossBound

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if args.visualize and args.visualize_outimages and args.steps_plot > 0 and step % args.steps_plot == 0 :
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'input step: {step}')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output  step: {step}')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output  step: {step}')
                board.image(color_transform(targets[0].cpu().data),
                    f'target step: {step}')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        scheduler.step()    ## scheduler 2    
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        with torch.no_grad():
            for step, (images, labels) in enumerate(loader_val):
                start_time = time.time()
                if args.cuda:
                    images = images.to(args.device)
                    labels = labels.to(args.device)

                inputs = Variable(images)    
                targets = Variable(labels)
                if args.embed!=0:
                    _,outputs,_ = model(inputs) 
                else:
                    _,outputs = model(inputs) 

                loss = criterion(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())
                time_val.append(time.time() - start_time)


                if (doIouVal):
                    iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

                if args.visualize and args.visualize_outimages and args.steps_plot > 0 and step % args.steps_plot == 0:
                    start_time_plot = time.time()
                    image = inputs[0].cpu().data
                    board.image(image, f'VAL input step: {step})')
                    if isinstance(outputs, list):  
                        board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                        f'VAL output step: {step}')
                    else:
                        board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                        f'VAL output , step: {step}')
                    board.image(color_transform(targets[0].cpu().data),
                        f'VAL target step: {step})')
                    print ("Time to paint images: ", time.time() - start_time_plot)
                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
                    print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                            "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
            if args.visualize:
                board.add_scalar(win=f'Validation IoU {args.model} {pretrained} {args.knowledge_distillation_feature_fusion} {args.knowledge_distillation_loss}', x=epoch, y=iouVal)
           

        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        
        filenameCheckpoint = savedir + f'/checkpoint_{savefile}.pth.tar'
        filenameBest = savedir + f'/model_best_{savefile}.pth.tar'    

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH

        filename = f'{savedir}/model_{savefile}-{epoch:03}.pth'
        filenamebest = f'{savedir}/model_{savefile}_best.pth'

        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            with open(savedir + f"/best_{savefile}.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))
                myfile.write(class_iou_messages(iou_classes))   

        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
        if args.visualize:    
            board.add_doubleline(epoch=epoch, val_loss=average_epoch_loss_val, train_loss=average_epoch_loss_train, title=f'Losses {pretrained} {args.knowledge_distillation_feature_fusion} {args.knowledge_distillation_loss}', win= f'Losses {args.model}')
   
    return model   

def class_iou_messages(iou_classes):
    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)
    classes = [
        'background',  # 0
        'aeroplane',  # 1
        'bicycle',  # 2
        'bird',  # 3
        'boat',  # 4
        'bottle',  # 5
        'bus',  # 6
        'car',  # 7
        'cat',  # 8
        'chair',  # 9
        'cow',  # 10
        'diningtable',  # 11
        'dog',  # 12
        'horse',  # 13
        'motorbike',  # 14
        'person',  # 15
        'pottedplant',  # 16
        'sheep',  # 17
        'sofa',  # 18
        'train',  # 19
        'tvmonitor'  # 20
    ]
    print_iou_classes = f'\nPer-Class IoU:'
    for i in range(iou_classes.size(0)):
        iou = iou_classes_str[i]
        classi = classes[i]
        print_iou_classes += f'\n{iou}\t{classi} '
    return print_iou_classes

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

def load_my_state_dict(model, state_dict): 
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


def main(args):
    if args.dataset == 'cityscapes':
        savedir = f'../save/Testbatch{args.batch_size}/Baseline'
    elif args.dataset == 'VOC':
        savedir = f'../datalogs/VOCSUGGER/distilla_b3_b0_2025_onlyembeds/dilltila_save_test/Testbatch{args.batch_size}-VOC/Baseline'
    else:
        assert 'Dataset does not exist'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    teacher = mit_b3()
    if args.dataset == 'cityscapes':
        teacher_path = '../outputs/segformerb2_teacher_cityscapes.pth'
    elif args.dataset == 'VOC':
        teacher_path = r'.\model_SegformerB3-ckpt-nonpretrained-2024-12-17_best.pth'
    else:
        assert 'dataset not supported'

    if args.model.startswith('Segformer'):
        if args.model.endswith('B0'): 
            model = mit_b0(num_classes=NUM_CLASSES)
            path = r'/'
        elif args.model.endswith('B1'):
            model = mit_b1()
            path = '/ckpt_pretrained/mit_b1.pth'  
        if args.student_pretrained:   
            print('load weights from pretrained ckpt ...')               
            save_model = torch.load(path)
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict, strict=False)
        else:
            print('no pretrained ckpt loaded ...')
    else: 
        assert 'unsupported model'

    model = FEF(model, args.embed)
    if args.continue_path is not None:
        continue_path = args.continue_path
        model = load_my_state_dict(model, torch.load(continue_path))
        print('load continue path:', continue_path)
    print("teacher_path:", teacher_path)

    total_parameters, total_flops = netParams(model, (2, 3, 512, 512))
    print("参数数量: %d ==> %.2f M" % (total_parameters, (total_parameters / 1e6)))
    print("计算量（FLOPs）: %.2f M" % (total_flops / 1e6))

    if args.cuda:
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            teacher = load_my_state_dict(teacher,torch.load(teacher_path))#True
            model=model.to(args.device)
            teacher = teacher.to(args.device)
            model = nn.DataParallel(model)  
            teacher = nn.DataParallel(teacher)
        else:
            # args.gpu_nums = 1
            print("single GPU for training")
            teacher = load_my_state_dict(teacher,torch.load(teacher_path))
            model = model.to(args.device) 
            teacher = teacher.to(args.device)


    print("========== STUDENT TRAINING ===========")
    model = train(args, model, teacher) #Train encoder

    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True) 
    parser.add_argument('--model', default="SegformerB0")

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--dataset',default="VOC", choices=['cityscapes', "VOC"])
    parser.add_argument('--datadir', default=r".\VOC2007")
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)  
    parser.add_argument('--savedate', default=True)
    parser.add_argument('--visualize', default=False)
    parser.add_argument('--visualize_outimages', action='store_true',default=False)
    parser.add_argument('--iouTrain', action='store_true', default=False) 
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument("--device", default='cuda', help="Device on which the network will be trained. Default: cuda")
    parser.add_argument("--knowledge-distillation-feature-fusion", default= 'CSF', choices=['CSF'])
    parser.add_argument("--knowledge-distillation-loss", default="MultiScaleContextAlignmentDistillationLoss", choices= ['MultiScaleContextAlignmentDistillationLoss', 'SelfSupervisedDistillationLoss','FeatureAlignmentLoss'])
    parser.add_argument('--kd-tau', default=1., type=float, help="")
    parser.add_argument('--kd-alpha', default=0.5, type=float, help="")
    parser.add_argument('--student-pretrained',default=False)
    parser.add_argument('--review-kd-loss-weight', type=float, default=1.0,
                    help='feature konwledge distillation loss weight')
    parser.add_argument('--teacher-val', default= True, help='validate teacher before training process')
    parser.add_argument("--temperature", type=float, default=1.0, help="normalize temperature")
    parser.add_argument("--lambda-cwd", type=float, default=1.0, help="lambda_kd")
    parser.add_argument("--norm-type", type=str, default='channel',choices=['channel','spatial','channel_mean'], help="kd normalize setting")#不改
    parser.add_argument("--divergence", type=str, default='kl', help="kd divergence setting")
    parser.add_argument("--embed",type=int,default=5, choices=[0,1,2,3,4,5])
    parser.add_argument('--continue-path',default=None)
    parser.add_argument('--lr',type=float,default=3e-5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)


    main(parser.parse_args())
