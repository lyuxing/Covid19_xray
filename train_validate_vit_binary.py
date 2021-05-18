# MURA dataset classification
# Author: Xing Lu
# Version: 0 @ 2020-11-18 11:41:43
# Notes: baseline for training.

import torch.nn as nn
from torch import optim
from time import gmtime, strftime
from tensorboardX import SummaryWriter
from tqdm import tqdm

import timm

# from model.resnet2d_cbam import *
# from model.resnet2d import *
from data_generator_imgaug import *
from tools import *
from Own_util.Visualizations import Visualizations
from lr_cosine_wm import CosineAnnealingWarmupRestarts
from sampler import BalancedBatchSampler
from focal_loss import *

import torchvision

import warnings
warnings.filterwarnings(action='ignore')



if __name__ == '__main__':
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    # result_path = r'E:\Data\HeronTech\SDFY\ori_marigin_crop\model_train'
    result_path = r'E:\Xing\Covid_19_xray\train_log'
    descrip = 'Apr27_timm_vit_224_imgaug_binary_bce'
    debug = True
    model_save_path = os.path.join(result_path, descrip, time_string, 'save')
    tb_save_path = os.path.join(result_path, descrip,time_string, 'tb')
    os.makedirs(model_save_path)
    os.makedirs(tb_save_path)
    writer = SummaryWriter(log_dir=tb_save_path)


    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


    num_classes = 2

    # get model from custom design
    # model = resnet18(pretrained=True,num_classes=num_classes)
    visual = Visualizations(class_names=list(np.arange(num_classes)),img_num=16)

    # get model from timm
    # model = timm.create_model('gluon_resnet34_v1b', pretrained=True)
    # model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    # model = timm.create_model('swin_base_patch4_window12_384', pretrained=True,num_classes = 2)
    # model = timm.create_model('vit_base_patch16_384', pretrained=True,num_classes = 2)
    # img_size = 384
    # train_bs = 8
    # valid_bs = 8

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)
    # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=1)
    # model = timm.create_model('vit_base_resnet50d_224', pretrained=True, num_classes=2)
    img_size = 224
    train_bs = 32
    valid_bs = 8

    # get model from pretrained packages
    # model = torchvision.models.resnet34(pretrained=True)
    # model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
    # model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

    if torch.cuda.device_count()>1:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    criterion = nn.BCELoss().cuda()
    # criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss().cuda()

    lr = 1e-3

    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold=0.01, factor=0.3)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=6, cycle_mult=2.0, max_lr=lr, min_lr=1e-8, warmup_steps=2, gamma=0.5)



    # Creating a Transformation Object
    # train_transform = transforms.Compose([
    #     # Converting images to the size that the model expects
    #     # torchvision.transforms.Resize(size=(224, 224)),
    #     transforms.RandomResizedCrop(size=(img_size, img_size),scale=(0.7,1.1),ratio=(0.7,1.1)),
    #     transforms.RandomHorizontalFlip(),  # A RandomHorizontalFlip to augment our data
    #     # torchvision.transforms.RandomVerticalFlip(),  # A RandomHorizontalFlip to augment our data
    #     transforms.RandomAffine(degrees=(-15, 15)),
    #     transforms.ColorJitter(brightness=[0.7, 1.1], contrast=[0.7, 1.1]),
    #     transforms.ToTensor(),  # Converting to tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    #     # Normalizing the data to the data that the ResNet18 was trained on
    #
    # ])


    # Creating a Transformation Object
    test_transform = torchvision.transforms.Compose([
        # Converting images to the size that the model expects
        # torchvision.transforms.Resize(size=(img_size, img_size)),
        # We don't do data augmentation in the test/val set
        torchvision.transforms.ToTensor(),  # Converting to tensor
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Normalizing the data to the data that the ResNet18 was trained on

    ])



    trainconfig = {"dataset": 'mura',"subset": '0'}
    train_config = dataconfig(**trainconfig)
    training_data = DataGenerator(train_config,transform= test_transform)
    # train_loader = DataLoader(training_data, num_workers=4, batch_size=train_bs, shuffle= True)
    train_loader = DataLoader(training_data,num_workers=4, sampler=BalancedBatchSampler(training_data, type='single_label'), batch_size = train_bs)

    valconfig = {"dataset": "mura","subset": '1'}
    val_config = dataconfig(**valconfig)
    validation_data = DataGenerator(val_config,transform= test_transform)
    val_loader = DataLoader(validation_data, num_workers=4,batch_size=valid_bs, drop_last=True)

    print('data loader finished')

    Train_C_flag = False
    epoch_len = 300

    bst_acc = 0
    bst_loss = 5
    bst_tsh = 0.1
    better_epoch = 0


    if Train_C_flag == True:
        model_load_path = r'E:\Xing\mass0508\Train_log\June1_resnet50_cbam_dp_marigin_balancesampler_lr\Tue02Jun2020-065119\save'
        model_name = r'\best_model.pth'

        # model_load_path = r'pretrain'
        # model_name = r'\resnet_34.pth'

        checkpoint = torch.load(model_load_path + model_name)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        Epoch = checkpoint['epoch']
    else:
        Epoch = 0

    for epoch in range(Epoch,Epoch+epoch_len):
        model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()

        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        for i, (images,labels) in tqdm(enumerate(train_loader)):
            targets = labels.float().reshape(-1,1).cuda()
            # targets = labels.cuda()
            outputs = model(images.cuda())
            outputs = F.sigmoid(outputs)
            # outputs = F.softmax(outputs)
            # print('outputs: ', outputs.data.cpu().numpy().tolist(), 'targets: ', targets.data.cpu().numpy().tolist())


            # loss = criterion(outputs, targets.long())
            loss = criterion(outputs, targets)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # acc = calculate_accuracy(outputs, targets)
            acc = calculate_accuracy_binary(outputs, targets)
            losses.update(loss.item(), targets.size(0))
            accuracies.update(acc, targets.size(0))

            if (epoch) % 10 ==0 and i % 200 == 0:
                # _, predict = torch.max(outputs, 1)
                predict = outputs > 0.5
                if debug:
                    visual.show_images(images, labels, predict,save_path= tb_save_path,title='train_{}_{}'.format(epoch,i))
                    # visual.show_cam_images(model,255*images.cuda(), labels.cuda(), predict,criterion,frac=0.6)
                # predict = outputs>0.5
                add_image_3d(images, predict, targets, writer, column = np.sqrt(train_bs),subset='train', epoch=epoch, name= str(i)+'_image')


        losses_val = AverageMeter()
        accuracies_val = AverageMeter()
        model.eval()
        with torch.no_grad():
            for j, (inputs_val, labels_val) in enumerate(val_loader):
                targets_val = labels_val.float().reshape(-1,1).cuda()
                outputs_val = model(inputs_val.cuda())
                outputs_val = F.sigmoid(outputs_val)
                # outputs_val = F.softmax(outputs_val)
                loss_val = criterion(outputs_val, targets_val)
                acc_val = calculate_accuracy_binary(outputs_val, targets_val)
                # acc_val = calculate_accuracy(outputs_val, targets_val)
                losses_val.update(loss_val.item(), targets_val.size(0))
                accuracies_val.update(acc_val, targets_val.size(0))

                if (epoch ) % 10 == 0 and j % 10 == 0:
                    # print(j, loss_val)
                    # _,predict = torch.max(outputs_val,1)
                    predict = outputs_val > 0.5
                    if debug:
                        visual.show_images(inputs_val, labels_val, predict,save_path= tb_save_path, title='valid_{}_{}'.format(epoch,j))

                    add_image_3d(inputs_val, predict, targets_val, writer, column = np.sqrt(valid_bs), subset='val', epoch=epoch, name= str(i)+'_image')

        # scheduler.step(losses_val.avg)
        scheduler.step()

        print('epoch: ', epoch+1, 'train_loss: ', losses.avg, 'train_acc: ', accuracies.avg,
              'val_loss: ', losses_val.avg, 'val_acc: ', accuracies_val.avg)


        if epoch >3 and bst_acc <= accuracies_val.avg :

            save_file_path = os.path.join(model_save_path, 'best_model.pth')
            states = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(states, save_file_path)
            better_epoch = epoch

            bst_acc = accuracies_val.avg
            bst_loss = losses_val.avg

        print('better model found at epoch {} with val_loss {} and val_acc {}'.format(better_epoch,bst_loss,bst_acc))

        # Save model and print something in the tensorboard
        # if not debug:
        writer.add_scalars('loss/epoch',
                           {'train loss': losses.avg, 'validation loss': losses_val.avg}, epoch + 1)
        writer.add_scalars('acc/epoch',
                           {'train accuracy': accuracies.avg, 'validation accuracy': accuracies_val.avg}, epoch + 1)
        writer.add_scalars('Learning Rate/epoch',
                               {'train accuracy': optimizer.param_groups[0]['lr']}, epoch + 1)


