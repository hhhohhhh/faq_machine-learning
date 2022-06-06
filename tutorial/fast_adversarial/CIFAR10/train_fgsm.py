import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from tutorial.fast_adversarial.CIFAR10.preact_resnet import PreActResNet18
from tutorial.fast_adversarial.CIFAR10.utils_fast_adversarial import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)



# logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='corpus/cifar-data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='output/train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    """
    其中的opt-level参数是用户指定采用何种数据格式做训练的。
        O0：纯FP32训练，可以作为accuracy的baseline；
        O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
        O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算。
        O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；
    """
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    from utils.common import init_logger
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logger = init_logger(output_dir=args.out_dir,log_filename='output.log')

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    # 约束的半径
    epsilon = (args.epsilon / 255.) / std
    # 对抗样本生成的 步长， 当 alpha= 1.25 epsilon，论文中得到最优的结果
    alpha = (args.alpha / 255.) / std
    # pgd 对抗样本生成的 步长
    pgd_alpha = (2 / 255.) / std

    model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    # opt_level ='02'
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    # 使用 amp 初始化模型
    model, opt = amp.initialize(models=model, optimizers=opt, **amp_args)
    # 定义 loss
    criterion = nn.CrossEntropyLoss()

    # 初始化 对抗样本的参数
    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)

    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            # 随机初始化
            if args.delta_init == 'random':
                # 对 每个 channel 进行初始化
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                # 进行最大最小的裁剪
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            # R+FGSM Training相当于两次ST（每次计算两次梯度，一个用于扰动生成，一个用于更新模型参数）
            # 对抗样本的生成的带约束的优化问题  : x_adversarial = argmax( I(f(x') !=y))), where x'-x<=epsilon
            # 首先加入 对抗的扰动到原始的输入 X -> X + delta[:X.size(0)]
            output = model(X + delta[:X.size(0)])
            # 使用 cross_entropy 替代 0-1​损失
            loss = F.cross_entropy(output, y)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            # 使用对抗样本的约束条件限制
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            # 生成对抗样本
            delta = delta.detach()

            # 将 对抗样本 加入原始输入中重新计算输出来更新模型参数
            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
