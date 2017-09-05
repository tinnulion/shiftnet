import os
import sys
import time
import argparse
import numpy
import torch
import torch.nn
import torchvision.transforms
import torch.utils

DEVICE_ID = 0

IMAGE_SIZE = 64
CROP_SIZE = 56
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
NUM_EPOCHS = 100
PRINT_FREQ = 1

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001


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


def get_lr(epoch):
    if epoch < 60:
        return 0.01
    elif epoch < 90:
        return 0.001
    else:
        return 0.0001


def adjust_lr(optimizer, epoch):
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    batch_size = target.size

    top_args = numpy.argsort(output, axis=1)
    top1 = top_args[:, 0]
    top5 = top_args[:, :5]
    pred = numpy.tile(target, (5, 1)).T

    diff_top1 = target == top1
    diff_top5 = numpy.sum(pred == top5, axis=1)
    acc_top1 = numpy.count_nonzero(diff_top1)
    acc_top5 = numpy.count_nonzero(diff_top5)

    acc_top1 /= float(batch_size)
    acc_top5 /= float(batch_size)

    return acc_top1, acc_top5


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_cuda = input.cuda(device=DEVICE_ID, async=True)
        target_cuda = target.cuda(device=DEVICE_ID, async=True)
        input_var = torch.autograd.Variable(input_cuda, requires_grad=True)
        target_var = torch.autograd.Variable(target_cuda)

        output = model(input_var)
        output_cpu = output.data.cpu().numpy()
        loss = criterion(output, target_var)
        loss_cpu = loss.data.cpu().numpy()
        target_cpu = target.numpy()

        prec1, prec5 = accuracy(output_cpu, target_cpu)
        losses.update(loss_cpu[0], input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Epoch: [{:04d}] [{:04d}/{:04d}]\t'
                  'Time (sec) = {batch_time.val:>7.3f}, avg {batch_time.avg:<7.3f}\t'
                  'Data (sec) = {data_time.val:>7.3f}, avg {data_time.avg:<7.3f}\t'
                  'Loss = {loss.val:>7.4f}, avg {loss.avg:<7.4f}\t'
                  'Acc-1 = {top1.val:>.3f}, avg {top1.avg:<.3f}\t'
                  'Acc-5 = {top5.val:>.3f}, avg {top5.avg:<.3f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        input_cuda = input.cuda(device=DEVICE_ID, async=True)
        target_cuda = target.cuda(device=DEVICE_ID, async=True)
        input_var = torch.autograd.Variable(input_cuda, volatile=True)
        target_var = torch.autograd.Variable(target_cuda, volatile=True)

        output = model(input_var)
        output_cpu = output.data.cpu().numpy()
        loss = criterion(output, target_var)
        loss_cpu = loss.data.cpu().numpy()
        target_cpu = target.numpy()

        prec1, prec5 = accuracy(output_cpu, target_cpu)
        losses.update(loss_cpu[0], input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time (sec) = {batch_time.val:>7.3f}, avg {batch_time.avg:<7.3f}\t'
                  'Loss = {loss.val:>7.4f}, avg {loss.avg:<7.4f}\t'
                  'Acc-1 = {top1.val:>.3f}, avg {top1.avg:<.3f}\t'
                  'Acc-5 = {top5.val:>.3f}, avg {top5.avg:<.3f}'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))


def save_checkpoint(obj, path):
    torch.save(obj, path)
    print('Checkpoint saved to {}'.format(path))


if __name__ == "__main__":
    from pathlib import Path
    script_folder = str(Path(__file__).resolve().parents[0])

    parser = argparse.ArgumentParser(description='Tiny ImageNet training with ShiftNet')
    parser.add_argument('--train-dir', default='', type=str, help='Folder with train images', dest='train_dir')
    parser.add_argument('--val-dir', default='', type=str, help='Folder with val images', dest='val_dir')
    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        print('Cannot find train data at {}'.format(args.train_dir))
        sys.exit()
    if not os.path.exists(args.val_dir):
        print('Cannot find val data at {}'.format(args.val_dir))
        sys.exit()


    print("Preparing data loaders...")
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = torchvision.datasets.ImageFolder(
        args.train_dir,
        torchvision.transforms.Compose([
            torchvision.transforms.RandomSizedCrop(CROP_SIZE),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True)
    val_dataset = torchvision.datasets.ImageFolder(
        args.val_dir,
        torchvision.transforms.Compose([
            torchvision.transforms.Scale(IMAGE_SIZE),
            torchvision.transforms.CenterCrop(CROP_SIZE),
            torchvision.transforms.ToTensor(),
            normalize]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    print("Creating model, optimizer etc...")
    import shiftnet64
    model = shiftnet64.ShiftNet()
    model.cuda(device_id=DEVICE_ID)
    criterion = torch.nn.CrossEntropyLoss().cuda(device_id=DEVICE_ID)
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    print("Training...")
    for epoch in range(0, NUM_EPOCHS):
        adjust_lr(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        validate(val_loader, model, criterion)

        save_obj = {
            'arch': 'ShiftNet 56x56',
            'epochs_finished': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()}
        save_name = 'checkpoint_ep_{:04d}.pth'.format(epoch)
        save_path = os.path.join(script_folder, save_name)
        save_checkpoint(save_obj, save_path)

    print('Training done.')