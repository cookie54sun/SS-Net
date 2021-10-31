#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.slrsa_cls import get_model, cal_loss
import numpy as np
from torch.utils.data import DataLoader
from utils.util import IOStream
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import datetime


def train(args, io, time_now):
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, npoint=args.num_points, split='train',
                                       useNormalsAsLabels=args.normal)
    test_dataset = ModelNetDataLoader(root=data_path, npoint=args.num_points, split='test',
                                      useNormalsAsLabels=args.normal)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             drop_last=False, num_workers=4)

    device = torch.device("cuda" if args.cuda else "cpu")

    # load models
    model = get_model().to(device)
    print(str(model))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.load_model == True:
        io.cprint('Use pretrained model...')
        checkpoint = torch.load(args.model_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        io.cprint('Start training from scratch...')
        start_epoch = 0

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = torch.optim.Adam(
            model.parameters(),
            lr=args.lr_max,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss

    # use tensorboard to visualize loss and acc
    writer = SummaryWriter(log_dir=('checkpoints/' + args.exp_name+time_now + '/' + 'runs-'+time_now))

    best_test_acc = 0
    for epoch in range(start_epoch,args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        gamma = 1
        io.cprint('Epoch %d, lr: %.6f , gamma: %.6f' % (epoch, scheduler.get_lr()[0], gamma))
        train_loss = 0.0
        train_ACC_loss = 0.0
        train_Cos_loss = 0.0

        list_train_unique_num_points1 = []
        list_train_unique_num_points2 = []
        list_test_unique_num_points1 = []
        list_test_unique_num_points2 = []
        count = 0.0
        train_pred = []
        train_true = []
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)

        for batch_id, value in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9, ncols=60):
            model = model.train()
            data, label = value
            label = label[:, 0]
            data, label = data.to(device), label.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            logits, Cos_loss, train_unique_num_points1, train_unique_num_points2 = model(data, gamma=gamma, hard=True)
            ACC_loss = criterion(logits, label)
            Cos_loss = torch.mean(Cos_loss)*args.cos_loss_weight
            loss = ACC_loss + Cos_loss
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size

            train_ACC_loss += ACC_loss.item() * batch_size
            train_Cos_loss += Cos_loss.item() * batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            ## unique_num_points
            train_unique_num_points1 = torch.mean(train_unique_num_points1.type(torch.FloatTensor))
            train_unique_num_points2 = torch.mean(train_unique_num_points2.type(torch.FloatTensor))
            list_train_unique_num_points1.append(train_unique_num_points1.detach().cpu().numpy())
            list_train_unique_num_points2.append(train_unique_num_points2.detach().cpu().numpy())

        mean_train_unique_num_points1 = np.array(list_train_unique_num_points1).mean()
        mean_train_unique_num_points2 = np.array(list_train_unique_num_points2).mean()

        io.cprint('\nTrain %d, number of train sampled points in sa1:%.1f' % (epoch,mean_train_unique_num_points1))
        io.cprint('Train %d, number of train sampled points in sa2:%.1f' % (epoch,mean_train_unique_num_points2))
        writer.add_scalar('Points/train_sa1', mean_train_unique_num_points1, epoch)
        writer.add_scalar('Points/train_sa2', mean_train_unique_num_points2, epoch)

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        io.cprint('Train %d, loss: %.6f, Cos_Loss: %.6f' % (epoch,train_loss*1.0/count,
                                                            train_Cos_loss * 1.0 / count))

        io.cprint('Train %d, train acc: %.6f, train avg acc: %.6f' % (epoch,train_acc,train_avg_acc))

        writer.add_scalar('Loss/train', train_loss*1.0/count, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Avg_Acc/train', train_avg_acc, epoch)
        writer.add_scalar('ACC_Loss/train', train_ACC_loss * 1.0 / count, epoch)
        writer.add_scalar('Cos_Loss/train', train_Cos_loss * 1.0 / count, epoch)

        ####################
        # Test
        ####################
        with torch.no_grad():
            test_loss = 0.0
            test_ACC_loss = 0.0
            test_Cos_loss = 0.0
            count = 0.0
            test_pred = []
            test_true = []
            for data, label in test_loader:
                model = model.eval()
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]

                logits, Cos_loss, test_unique_num_points1, test_unique_num_points2 = model(data, gamma=gamma, hard=True)
                ACC_loss = criterion(logits, label)
                Cos_loss = torch.mean(Cos_loss)*args.cos_loss_weight
                loss = ACC_loss + Cos_loss
                preds = logits.max(dim=1)[1]
                count += batch_size

                test_loss += loss.item() * batch_size
                test_ACC_loss += ACC_loss.item() * batch_size
                test_Cos_loss += Cos_loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

                ## unique_num_points
                test_unique_num_points1 = torch.mean(test_unique_num_points1.type(torch.FloatTensor))
                test_unique_num_points2 = torch.mean(test_unique_num_points2.type(torch.FloatTensor))
                list_test_unique_num_points1.append(test_unique_num_points1.detach().cpu().numpy())
                list_test_unique_num_points2.append(test_unique_num_points2.detach().cpu().numpy())

        mean_test_unique_num_points1 = np.array(list_test_unique_num_points1).mean()
        mean_test_unique_num_points2 = np.array(list_test_unique_num_points2).mean()

        io.cprint('Test %d, number of test sampled points in sa1:%.1f' % (epoch,mean_test_unique_num_points1))
        io.cprint('Test %d, number of test sampled points in sa2:%.1f' % (epoch,mean_test_unique_num_points2))
        writer.add_scalar('Points/test_sa1', mean_test_unique_num_points1, epoch)
        writer.add_scalar('Points/test_sa2', mean_test_unique_num_points2, epoch)


        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_avg_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        io.cprint('Test %d, loss: %.6f, Cos_Loss: %.6f' % (epoch,test_loss*1.0/count,test_Cos_loss * 1.0 / count))

        io.cprint('Test %d, test acc: %.6f, test avg acc: %.6f' % (epoch,test_acc,test_avg_acc))
        writer.add_scalar('Loss/test', test_loss * 1.0 / count, epoch)
        writer.add_scalar('Acc/test', test_acc, epoch)
        writer.add_scalar('Avg_Acc/test', test_avg_acc, epoch)
        writer.add_scalar('ACC_Loss/test', test_ACC_loss * 1.0 / count, epoch)
        writer.add_scalar('Cos_Loss/test', test_Cos_loss * 1.0 / count, epoch)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            io.cprint('Saving model...')
            state = {
                'epoch': (epoch+1),
                'model_state_dict': model.state_dict(),
            }
            torch.save(state, 'checkpoints/%s/models/model-%f-%04d.pth' % (args.exp_name+time_now, test_acc, epoch))

    io.cprint('Best test acc : %.6f' % best_test_acc)
    writer.close()


def test(args, io):
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, npoint=args.num_points, split='test',
                                      useNormalsAsLabels=args.normal)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             drop_last=False, num_workers=4)
    device = torch.device("cuda" if args.cuda else "cpu")

    #load models
    model = get_model().to(device)
    model = nn.DataParallel(model)
    checkpoints = torch.load(args.model_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    model = model.eval()
    test_true = []
    test_pred = []
    gamma = 1
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits, Cos_loss, test_unique_num_points1, test_unique_num_points2 = model(data, gamma=gamma, hard=True)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


def _init_(time_now):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name+time_now):
        os.makedirs('checkpoints/'+args.exp_name+time_now)
    if not os.path.exists('checkpoints/'+args.exp_name+time_now+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+time_now+'/'+'models')

if __name__ == "__main__":
    # cal time
    start = time.time()

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='SLRSA_cls', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr_max', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='pre_trained/cls_model-0.927472-0194.pth', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--load_model', type=bool, default=False,
                        help='if load model when training (default: False)')
    parser.add_argument('--cos_loss_weight', type=float, default=0.01, # default=0.01,
                        help='wight of cosine loss (default: 1e-2)')
    parser.add_argument('--normal', type=bool, default=False,
                        help='if use normal as label (default: False)')
    args = parser.parse_args()

    time_now = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    _init_(time_now)

    io = IOStream('checkpoints/' + args.exp_name+time_now + '/run-'+time_now+'.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io, time_now)
    else:
        test(args, io)

    end = time.time()
    t = end-start
    print('training time used: ', t)

