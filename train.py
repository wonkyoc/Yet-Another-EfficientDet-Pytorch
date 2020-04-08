# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import datetime
import os
import argparse
import traceback

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from efficientdet.loss import FocalLoss
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights


class Params:
    """
    Here is what you have to manual specify when you train on a custom dataset.
    You need to prepare your dataset to be coco-styled

    put your dataset under datasets/ like this:

    datasets/
        -your_project_name/
            -train_set_name/
                -*.jpg
            -val_set_name/
                -*.jpg
            -annotations
                -instances_{train_set_name}.json
                -instances_{val_set_name}.json
    """
    project_name = 'coco'  # also the folder name of the dataset that under 'data_path' folder
    train_set = 'train2017'
    val_set = 'val2017'
    num_gpus = 4

    # mean and std in RGB order
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # must match your dataset's category_id.
    # category_id is one_indexed,
    # for example, index of 'car' here is 2, while category_id of is 3
    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=bool, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of epoches between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')

    args = parser.parse_args()
    return args


def train(opt):
    params = Params()

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    training_set = CocoDataset(root_dir=opt.data_path + params.project_name, set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(512 + opt.compound_coef * 128)]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=opt.data_path + params.project_name, set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(512 + opt.compound_coef * 128)]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_anchors=9, num_classes=len(params.obj_list), compound_coef=opt.compound_coef)

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0
        model.load_state_dict(torch.load(weights_path))
        print(f'loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    if params.num_gpus > 0:
        model = model.cuda()
        model = CustomDataParallel(model, params.num_gpus)

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    criterion = FocalLoss()

    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        try:
            model.train()
            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus > 0:
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    _, regression, classification, anchors = model(imgs)

                    cls_loss, reg_loss = criterion(classification, regression, anchors, annot,
                                                   imgs=imgs, obj_list=params.obj_list)

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                except Exception as e:
                    print(traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if step % opt.save_interval == 0 and step > 0:
                save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus > 0:
                            annot = annot.cuda()
                        _, regression, classification, anchors = model(imgs)
                        cls_loss, reg_loss = criterion(classification, regression, anchors, annot)

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch + 1, opt.num_epochs, cls_loss, reg_loss, loss.mean()))
                writer.add_scalars('Total_loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                    # onnx export is not tested.
                    # dummy_input = torch.rand(opt.batch_size, 3, 512, 512)
                    # if torch.cuda.is_available():
                    #     dummy_input = dummy_input.cuda()
                    # if isinstance(model, nn.DataParallel):
                    #     model.module.backbone_net.model.set_swish(memory_efficient=False)
                    #
                    #     torch.onnx.export(model.module, dummy_input,
                    #                       os.path.join(opt.saved_path, 'signatrix_efficientdet_coco.onnx'),
                    #                       verbose=False)
                    #     model.module.backbone_net.model.set_swish(memory_efficient=True)
                    # else:
                    #     model.backbone_net.model.set_swish(memory_efficient=False)
                    #
                    #     torch.onnx.export(model, dummy_input,
                    #                       os.path.join(opt.saved_path, 'signatrix_efficientdet_coco.onnx'),
                    #                       verbose=False)
                    #     model.backbone_net.model.set_swish(memory_efficient=True)

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, loss))
                    break
            writer.close()
        except KeyboardInterrupt:
            save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)