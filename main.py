import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from dataset.dataset import get_FF_dataset, get_FF_5_class
from builder.builder import Trainer
from evaluation import evaluate


def main(args):
    # --- Initialization --- 
    # init logger
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = SummaryWriter(args.output_dir)

    # --- Dataloader ---
    print("Loading dataset from {}".format(args.dataset_path))
    train_dataset = get_FF_5_class(args.dataset_path, mode='train', isTrain=True, img_size=args.image_size)
    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )
    
    valid_dataset = get_FF_dataset(args.dataset_path, mode='valid', isTrain=False, img_size=args.image_size, drop_rate=0.8)
    valid_dataloader = torch.utils.data.DataLoader(
        dataset = valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True
    )

    test_dataset = get_FF_dataset(args.dataset_path, mode='test', isTrain=False, img_size=args.image_size, drop_rate=0.8)
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True
    )

    # --- Model and Optimizer ---
    model = Trainer(args.backbone_name, args.use_mc, args.use_ncc, args.use_aim)
    if args.pretrained_backbone_path != '':
        msg = model.load_pretrained_backbone(path=args.pretrained_backbone_path)
        print("Found pretrained model at {} and loaded with msg: {}".format(args.pretrained_backbone_path, msg))
    model = model.cuda()

    # -- Warmup Setup --
    if args.warmup_iter > 0:
        for param in model.backbone.parameters():
                param.requires_grad = False

    optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=args.lr, betas=(0.9, 0.999))
    model.apply(to_sync_bn)
    model = nn.DataParallel(model)
    loss_func = nn.CrossEntropyLoss()
   
    # --- Train ---
    best_auc_valid= 0.
    for epoch in range(args.epochs):
        # in each epoch
        model.train()
        total_loss, total_num, total_acc = 0.0, 0, 0.0
        train_bar = tqdm(train_dataloader, desc='Train')
        for i, (img, label) in enumerate(train_bar):
            
            if (epoch * len(train_dataloader) + i) == args.warmup_iter:
                for param in model.module.backbone.parameters():
                    param.requires_grad = True
                optimizer.add_param_group({'params': model.module.backbone.parameters()})
                print("Backbone updated.")
            
            img = img.cuda()

            if not args.use_mc:
                label = model.module.convert_label_to_binary(label)
            label = label.cuda()

            score = model(img)
            pred = model.module.trans_to_eval(score)
            pred = pred.flatten().tolist()

            # loss
            loss = loss_func(score, label)
            total_loss += loss.item()

            # accuracy
            y_true, y_pred = np.array(label.cpu()), np.array(pred)
            y_true = np.where(y_true>=1, 1, 0)
            total_acc += accuracy_score(y_true, y_pred > 0.5)
            total_num += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            if i % args.loss_print_freq == 0:
                logger.add_scalar('train/loss', total_loss / total_num, epoch * len(train_dataloader) + i)
                logger.add_scalar('train/acc', total_acc / total_num, epoch * len(train_dataloader) + i)
                logger.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch * len(train_dataloader) + i)
                train_bar.set_description(desc  ="Train Epoch:{} Loss:{:.4f} Acc:{:.2f}".format(epoch, total_loss/total_num, total_acc/total_num))
                total_loss, total_num, total_acc = 0.0, 0, 0.0
                
            # valid / test
            if i % args.eval_print_freq == 0:
                valid_auc, valid_r_acc, valid_f_acc = evaluate(model, valid_dataloader)
                test_auc, test_r_acc, test_f_acc = evaluate(model, test_dataloader)
                logger.add_scalar('valid/auc', valid_auc, epoch * len(train_dataloader) + i)
                logger.add_scalar('valid/r_acc', valid_r_acc, epoch * len(train_dataloader) + i)
                logger.add_scalar('valid/f_acc', valid_f_acc, epoch * len(train_dataloader) + i)
                logger.add_scalar('test/auc', test_auc, epoch * len(train_dataloader) + i)
                logger.add_scalar('test/r_acc', test_r_acc, epoch * len(train_dataloader) + i)
                logger.add_scalar('test/f_acc', test_f_acc, epoch * len(train_dataloader) + i)
                model.train()

                if best_auc_valid < valid_auc:
                    best_auc_valid = valid_auc
                    model.module.save_ckpt(os.path.join(args.output_dir, 'best_auc_valid.pth'))

def to_sync_bn(mod):
    if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.SyncBatchNorm.convert_sync_batchnorm(mod)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FaceForensics Training')
    # -- log para --
    parser.add_argument('--output-dir', default='', type=str, help='Path to save logs and checkpoints')
    parser.add_argument('--loss-print-freq', default=50, type=int, help='Print loss every x iters')
    parser.add_argument('--eval-print-freq', default=1000, type=int, help='Print eval results every x iters')
    # -- train para --
    parser.add_argument('--epochs', default=5, type=int,
                        help='Number of total epochs of training')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        help='Initial learning rate', dest='lr')
    parser.add_argument('--backbone-name', type=str, default='resnet', help='Name of backbone module')
    parser.add_argument('--pretrained-backbone-path', default='', type=str, help='Path to pretrained backbone. Empty for training from scratch.')
    # -- dataset para --
    parser.add_argument('--image-size', default=256, type=int, help="Input image size (Default: 256)")
    parser.add_argument('-j', '--workers', default=12, type=int,
                        help='number of data loading workers (Default: 12)')
    parser.add_argument('--dataset-path', type=str, default='/path/to/FF++/', help='Dataset directory')
    parser.add_argument('--batch-size', default=32, type=int, help='Number of images in each mini-batch (Default: 32)')
    # -- method para -- 
    parser.add_argument('--use-ncc', action='store_true', help='Whether to use non-negative linear classifier (Default: false)')
    parser.add_argument('--use-aim', action='store_true', help='Whether to use augmented integration module (Default: false)')
    parser.add_argument('--use-mc', action='store_true', help='Whether to use multi-class supervision.')
    parser.add_argument('--warmup-iter', type=int, default=0, 
        help="Number of warmup iterations for AIM and Classifier. We recommend to use 20000 when AIM is involved.")
    args = parser.parse_args()
    
    main(args)