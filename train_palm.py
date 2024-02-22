import os
import argparse
import tqdm
import json
import re
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tensorboardX import SummaryWriter

from models import OmniglotModel
from omniglot import MetaOmniglotFolder, split_omniglot, ImageCache, transform_image, transform_label
from utils import find_latest_file

from mamldataset import MamlDataset


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def Variable_(tensor, *args_, **kwargs):
    '''
    Make variable cuda depending on the arguments
    '''
    # Unroll list or tuple
    if type(tensor) in (list, tuple):
        return [Variable_(t, *args_, **kwargs) for t in tensor]
    # Unroll dictionary
    if isinstance(tensor, dict):
        return {key: Variable_(v, *args_, **kwargs) for key, v in list(tensor.items())}
    # Normal tensor
    variable = Variable(tensor, *args_, **kwargs)
    if args.cuda:
        variable = variable.cuda()
    return variable


# Parsing
parser = argparse.ArgumentParser('Train reptile on PLAM')

# Mode
parser.add_argument('logdir', default="", help='Folder to store everything/load')

# - Training params
# parser.add_argument('--train_data', type=str, help='', default='../DAGAN/datasets/IITDdata_left_PSA2+DC+SC+W_6.npy')
# parser.add_argument('--test_data', type=str, help='', default='../DAGAN/datasets/IITDdata_right.npy')
parser.add_argument('--train_data', type=str, help='', default='../DAGAN/datasets/Tongji_session2_PSA2+DC+SC+W_6.npy')
parser.add_argument('--test_data', type=str, help='', default='../DAGAN/datasets/Tongji_session2.npy')
parser.add_argument('--n_way', type=int, help='n way', default=5)
parser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)  # default=1
parser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)  # 原15
parser.add_argument('--t_batchsz', type=int, help='train-batchsz', default=1000)
parser.add_argument('--imgsz', type=int, help='imgsz', default=28)  # 调节的图像尺寸
parser.add_argument('--imgc', type=int, help='imgc', default=1)
parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
parser.add_argument('--num_workers', type=int, help='数据加载子进程', default=0)

parser.add_argument('--meta-iterations', default=5000, type=int, help='number of meta iterations')
parser.add_argument('--start-meta-iteration', default=0, type=int, help='start iteration')
parser.add_argument('--iterations', default=5, type=int, help='number of base iterations')
parser.add_argument('--test-iterations', default=50, type=int, help='number of base iterations')
parser.add_argument('--batch', default=10, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=2., type=float, help='meta learning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')

# - General params
parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate-every', default=100, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--input', default='palm', help='Path to omniglot dataset')
parser.add_argument('--cuda', default=0, type=int, help='Use cuda')
parser.add_argument('--check-every', default=10000, type=int, help='Checkpoint every')
parser.add_argument('--checkpoint', default='',
                    help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')

# Do some processing
args = parser.parse_args()
print(args)
args_filename = os.path.join(args.logdir, 'args.json')
run_dir = args.logdir
check_dir = os.path.join(run_dir, 'checkpoint')

# By default, continue training
# Check if args.json exists
if os.path.exists(args_filename):
# if False:
    print('Attempting to resume training. (Delete {} to start over)'.format(args.logdir))
    # Resuming training is incompatible with other checkpoint
    # than the last one in logdir
    assert args.checkpoint == '', 'Cannot load other checkpoint when resuming training.'
    # Attempt to find checkpoint in logdir
    args.checkpoint = args.logdir
else:
    print('No previous training found. Starting fresh.')
    # Otherwise, initialize folders
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    # Write args to args.json

    # with open(args_filename, 'wb') as fp:
    with open(args_filename, 'w') as fp:

        json.dump(vars(args), fp, indent=4)

# Create tensorboard logger
logger = SummaryWriter(run_dir)

# Load data
# Resize is done by the MetaDataset because the result can be easily cached
# omniglot = MetaOmniglotFolder(args.input, size=(28, 28), cache=ImageCache(),
#                               transform_image=transform_image,
#                               transform_label=transform_label)
# meta_train, meta_test = split_omniglot(omniglot, args.validation)

# print('Meta-Train characters', len(meta_train))
# print('Meta-Test characters', len(meta_test))

train_data = MamlDataset(root=args.train_data,
                         mode='train', n_way=args.n_way,
                         k_shot=args.k_spt,
                         k_query=args.k_qry,
                         batchsz=args.t_batchsz,  #
                         imgsz=args.imgsz,
                         imgc=args.imgc)
test_data = MamlDataset(root=args.test_data, mode='test',
                        n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=100,
                        imgsz=args.imgsz,
                        imgc=args.imgc)

# Loss
cross_entropy = nn.NLLLoss()


def get_loss(prediction, labels):
    return cross_entropy(prediction, labels)


def do_learning(net, optimizer, x_spt, y_spt, task_num):
    net.train()
    for iteration in range(task_num):
        # Sample minibatch
        data, labels = x_spt[iteration], y_spt[iteration]
        # 这个10是args.batch
        # way 影响label的范围 shot中较大值影响label重复程度
        # data (10, 1, 28, 28) label tensor: (10,)] tensor([0,0,1,2,2,1,1,3,3,2])
        # 提取特征 Forward pass
        prediction = net(data)

        # Get loss
        loss = get_loss(prediction, labels)

        # Backward pass - Update fast net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return loss.data[0]
    return loss.data.item()


def do_evaluation(net, x_qry, y_qry, iterations):
    losses = []
    accuracies = []
    net.eval()
    for iteration in range(iterations):
        # Sample minibatch
        data, labels = x_qry[iteration], y_qry[iteration]
        # data (10,1, 28, 28) label (Tensor: (10,)}
        # Forward pass
        prediction = net(data)

        # Get loss
        loss = get_loss(prediction, labels)

        # Get accuracy
        argmax = net.predict(prediction)
        accuracy = (argmax == labels).float().mean()

        # losses.append(loss.data[0])
        losses.append(loss.data.item())
        # accuracies.append(accuracy.data[0])
        accuracies.append(accuracy.item())

    return np.mean(losses), np.mean(accuracies)


def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def log(mode,meta_loss,meta_accuracy,meta_lr):
    # (Logging)
    loss_ = '{}_loss'.format(mode)
    accuracy_ = '{}_accuracy'.format(mode)
    meta_lr_ = 'meta_lr'
    info.setdefault(loss_, {})
    info.setdefault(accuracy_, {})
    info.setdefault(meta_lr_, {})
    info[loss_][meta_iteration] = meta_loss
    info[accuracy_][meta_iteration] = meta_accuracy
    info[meta_lr_][meta_iteration] = meta_lr
    print('\nMeta-{}'.format(mode))
    print('average metaloss', np.mean(list(info[loss_].values())))
    print('average accuracy', np.mean(list(info[accuracy_].values())))
    logger.add_scalar(loss_, meta_loss, meta_iteration)
    logger.add_scalar(accuracy_, meta_accuracy, meta_iteration)
    logger.add_scalar(meta_lr_, meta_lr, meta_iteration)
    return


# Build model, optimizer, and set states
meta_net = OmniglotModel(args.n_way, args.imgc, args.imgsz)
if args.cuda:
    meta_net.cuda()
meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr)
info = {}
state = None

# checkpoint is directory -> Find last model or '' if does not exist
if os.path.isdir(args.checkpoint):
    latest_checkpoint = find_latest_file(check_dir)
    if latest_checkpoint:
        print('Latest checkpoint found:', latest_checkpoint)
        args.checkpoint = os.path.join(check_dir, latest_checkpoint)
    else:
        args.checkpoint = ''

# Start fresh
if args.checkpoint == '':
    print('No checkpoint. Starting fresh')

# Load file
elif os.path.isfile(args.checkpoint):
    print('Attempting to load checkpoint', args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    meta_net.load_state_dict(checkpoint['meta_net'])
    meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
    state = checkpoint['optimizer']
    args.start_meta_iteration = checkpoint['meta_iteration']
    info = checkpoint['info']
else:
    raise ArgumentError('Bad checkpoint. Delete logdir folder to start over.')

# Main loop
for meta_iteration in tqdm.trange(args.start_meta_iteration, args.meta_iterations):
    db = DataLoader(train_data, args.task_num, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)  # 生成可以将所有任务跑一遍的迭代器

    # Update learning rate
    meta_lr = args.meta_lr * (1. - meta_iteration / float(args.meta_iterations))
    set_learning_rate(meta_optimizer, meta_lr)

    # Clone model
    net = meta_net.clone()
    optimizer = get_optimizer(net, state)
    # load state of base optimizer?

    # 训练数据 Sample base task from Meta-Train
    (x_spt, y_spt, x_qry, y_qry) = next(iter(db))
    # Update fast net
    loss = do_learning(net, optimizer, x_spt, y_spt, args.task_num)
    state = optimizer.state_dict()  # save optimizer state

    # Update slow net
    meta_net.point_grad_to(net)
    meta_optimizer.step()

    # Meta-Evaluation
    if meta_iteration % args.validate_every == 0:
        print('\n\nMeta-iteration', meta_iteration)
        print('(started at {})'.format(args.start_meta_iteration))
        print('Meta LR', meta_lr)

        (x_spt, y_spt, x_qry, y_qry) = next(iter(db))
        # Base-train
        net = meta_net.clone()
        optimizer = get_optimizer(net, state)  # do not save state of optimizer
        loss = do_learning(net, optimizer, x_spt, y_spt, args.task_num)

        # Base-test: compute meta-loss, which is base-validation error
        meta_loss, meta_accuracy = do_evaluation(net, x_qry, y_qry, 1)  # only one iteration for eval
        log("train",meta_loss,meta_accuracy,meta_lr)

        db_test = DataLoader(test_data, 1, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)  # 测试 生成可以将所有任务跑一遍的迭代器
        (x_spt, y_spt, x_qry, y_qry) = next(iter(db_test))
        # Base-train
        net = meta_net.clone()
        optimizer = get_optimizer(net, state)  # do not save state of optimizer
        loss = do_learning(net, optimizer, x_spt, y_spt, 1)
        # Base-test: compute meta-loss, which is base-validation error
        meta_loss, meta_accuracy = do_evaluation(net, x_qry, y_qry, 1)

        log("val",meta_loss,meta_accuracy,meta_lr)

    if meta_iteration % args.check_every == 0 and not (args.checkpoint and meta_iteration == args.start_meta_iteration):
        # Make a checkpoint
        checkpoint = {
            'meta_net': meta_net.state_dict(),
            'meta_optimizer': meta_optimizer.state_dict(),
            'optimizer': state,
            'meta_iteration': meta_iteration,
            'info': info
        }
        checkpoint_path = os.path.join(check_dir, 'check-{}.pth'.format(meta_iteration))
        torch.save(checkpoint, checkpoint_path)
        print('Saved checkpoint to', checkpoint_path)
