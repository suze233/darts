import os
import sys
import time
import glob
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

parser = argparse.ArgumentParser("cora")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=1, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

# todo
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CORA_CLASSES = 7


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    dataset = Planetoid(root=args.data, name='Cora')
    train_data = dataset[0].cuda()  # 获取图数据
    in_channels = dataset.num_features
    hidden_channels = args.init_channels
    out_channels = dataset.num_classes

    model = Network(args.init_channels, CORA_CLASSES, args.layers, criterion, in_channels, hidden_channels, out_channels)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # 设置w的优化器
    optimizer = torch.optim.SGD(
        model.parameters(),  # 优化器更新的参数，这里更新的是w
        args.learning_rate,  # 初始值是0.025，使用的余弦退火调度更新学习率，每个epoch的学习率都不一样
        momentum=args.momentum,  # 0.9
        weight_decay=args.weight_decay  # 正则化参数3e-4
    )


    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))

    train_idx = train_data.train_mask.nonzero(as_tuple=True)[0].cuda()
    valid_idx = train_data.val_mask.nonzero(as_tuple=True)[0].cuda()

    '''
    CosineAnnealingLR是余弦退火学习率调度器, 动态调整学习率
    optimizer: 优化器, 这里是w的优化器
    '''
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.epochs,
        eta_min=args.learning_rate_min
    )

    # 创建架构
    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]  # 得到本次迭代的学习率lr
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()  # 对应论文2.4 选出来权重值大的两个前驱节点，并把(操作，前驱节点)存下来
        logging.info('genotype = %s', genotype)
        # todo
        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_auc, train_obj = train(
            data=train_data,
            train_idx=train_idx,
            valid_idx=valid_idx,
            model=model,
            architect=architect,
            criterion=criterion,
            optimizer=optimizer,  # w的优化器
            lr=lr  # 当前epoch的学习率
        )
        logging.info('train_acc %f', train_auc)

        # validation
        valid_auc, valid_obj = infer(train_data, valid_idx, model, criterion)
        logging.info('valid_acc %f', valid_auc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(data, train_idx, valid_idx, model, architect, criterion, optimizer, lr):
    """
    对应伪代码的第一步和第二步
    :param data: 全部数据
    :param train_idx: 训练集索引
    :param valid_idx: 验证集索引
    :param model: 模型
    :param architect: 架构
    :param criterion: 损失函数
    :param optimizer: w的优化器
    :param lr: 学习率
    :return: top1正确率，loss
    """
    data = data.cuda()
    model = model.cuda()

    objs = utils.AvgrageMeter()  # 保存loss
    auc = utils.AvgrageMeter()  # auc

    model.train()

    # 对α进行更新，对应伪代码的第一步 公式6
    architect.step(
        data=data,
        train_idx=train_idx,
        valid_idx=valid_idx,
        eta=lr,
        network_optimizer=optimizer,  # w的优化器
        unrolled=args.unrolled
    )

    optimizer.zero_grad()  # 清除之前学到的梯度的参数

    # 对w进行更新，对应伪代码的第二步
    logits = model(data)
    loss = criterion(logits[train_idx], data.y[train_idx])  # 使用预测值logits和真实值target计算loss
    loss.backward()  # 反向传播，计算梯度（w）

    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
    optimizer.step()  # 应用梯度

    _auc = utils.eval_acc(out=logits[train_idx], label=data.y[train_idx])

    # todo n是什么
    objs.update(loss.data.item())
    auc.update(_auc)

    logging.info('train %f %f',  objs.avg, auc.avg)


    # for step, (input, target) in enumerate(train_queue):  # 每个step取出一个batch，batchsize是64（256个数据对）
    #     model.train()
    #     n = input.size(0)
    #
    #     input = input.cuda()  # requires_grad默认为False，不对input求导
    #     target = target.cuda(non_blocking=True)  # 使用non_blocking=True代替async=True
    #
    #     # 更新α是用validation set进行更新的，所以我们每次都从valid_queue拿出一个batch传入architect.step()
    #     # 用于alpha更新的一个batch 。使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问；
    #     input_search, target_search = next(iter(valid_queue))  # 从验证集中取
    #     input_search = input_search.cuda()
    #     target_search = target_search.cuda()
    #
    #     # 对α进行更新，对应伪代码的第一步 公式6
    #     architect.step(
    #         input_train=input,
    #         target_train=target,
    #         input_valid=input_search,
    #         target_valid=target_search,
    #         eta=lr,
    #         network_optimizer=optimizer,  # w的优化器
    #         unrolled=args.unrolled
    #     )
    #
    #     optimizer.zero_grad()  # 清除之前学到的梯度的参数
    #
    #     # 对w进行更新，对应伪代码的第二步
    #     logits = model(input)  # input来自训练集
    #     loss = criterion(logits, target)  # 使用预测值logits和真实值target计算loss
    #     loss.backward()  # 反向传播，计算梯度（w）
    #
    #     nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
    #     optimizer.step()  # 应用梯度
    #
    #     prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    #     objs.update(loss.data.item(), n)
    #     top1.update(prec1.data.item(), n)
    #     top5.update(prec5.data.item(), n)
    #
    #     if step % args.report_freq == 0:
    #         logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return auc.avg, objs.avg

# 只前向传播，计算loss
def infer(data, valid_idx, model, criterion):
    data = data.cuda()
    objs = utils.AvgrageMeter()
    auc = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        # input = input.cuda()
        # target = target.cuda()

        logits = model(data)
        loss = criterion(logits[valid_idx], data.y[valid_idx])

        _auc = utils.eval_acc(out=logits[valid_idx], label=data.y[valid_idx])

        # n = input.size(0)
        # todo
        objs.update(loss.data.item())
        auc.update(_auc)

        logging.info('valid %f %f', objs.avg, auc.avg)

    return auc.avg, objs.avg


if __name__ == '__main__':
    main()
