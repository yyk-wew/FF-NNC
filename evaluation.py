import torch
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import auc as cal_auc
from tqdm import tqdm

def cal_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true,y_pred,pos_label=1)
    auc_score = cal_auc(fpr, tpr)

    idx_real = np.where(y_true==0)[0]
    idx_fake = np.where(y_true==1)[0]

    if idx_real.shape[0] > 0:
        r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    else:
        r_acc = 0.

    if idx_fake.shape[0] > 0:
        f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)
    else:
        f_acc = 0.

    return auc_score, r_acc, f_acc
        

def evaluate(model, dataloader):
    # Attention: CUDA is used
    model.eval()
    # prediction
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in tqdm(dataloader, desc='Evaluation'):
            img = img.cuda()
            label = label.cuda()
            output = model.forward(img)
            y_pred.extend(output.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    auc_score, r_acc, f_acc = cal_score(y_true, y_pred)

    return auc_score, r_acc, f_acc


if __name__ == "__main__":
    from dataset.dataset import get_FF_dataset, get_test_dataset
    from builder.builder import Trainer
    import torch.nn as nn
    import argparse
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--ckpt-path', default='', type=str, help='Path to model checkpoints')
    parser.add_argument('--dataset-path', type=str, default='/data/yike/FF++_std_c23_300frames/', help='Location of dataset')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
    parser.add_argument('--batch-size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--backbone-name', type=str, default='resnet', help='Name of backbone module')
    parser.add_argument('--image-size', default=256, type=int, help="Input image size (default: 256)")
    parser.add_argument('--use-ncc', action='store_true', help='Whether to use non-negative linear classifier (Default: false)')
    parser.add_argument('--use-aim', action='store_true', help='Whether to use augmented integration module (Default: false)')
    parser.add_argument('--use-mc', action='store_true', help='Whether to use multi-class supervision.')
    
    args = parser.parse_args()
    
    test_dataset = get_FF_dataset(args.dataset_path, mode='test', isTrain=False, img_size=args.image_size)
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True
    )

    model = Trainer(args.backbone_name, use_mc=args.use_mc, use_ncc=args.use_ncc, use_aim=args.use_aim)
    model.load_state_dict(torch.load(args.ckpt_path)['model'], strict=True)
    model = model.cuda()
    model = nn.DataParallel(model)

    auc_score, r_acc, f_acc = evaluate(model, test_dataloader)

    print(auc_score, r_acc, f_acc)