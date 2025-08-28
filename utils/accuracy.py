import time
import torch
from loguru import logger

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Resets the meter to initial values.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The new value to update the meter with.
            n (int, optional): The number of values to update. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """
        Returns a string representation of the meter.
        """
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """
    Displays the progress of training or validation.
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """
        Displays the progress of the current batch.

        Args:
            batch (int): The current batch number.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """
        Gets the format string for the batch display.

        Args:
            num_batches (int): The total number of batches.

        Returns:
            str: The format string for the batch display.
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): The output predictions from the model.
        target (torch.Tensor): The ground truth labels.
        topk (tuple, optional): The top k values to consider. Defaults to (1,).

    Returns:
        list: A list of accuracies for the specified top k values.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    """
    Validates a model using a validation dataset.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The model to be validated.
        device (torch.device, optional): The device to run the model on. If None, uses the device of the first model parameter. Defaults to None.
        print_freq (int, optional): Frequency of printing progress. Defaults to 100.

    Returns:
        float: Top-1 accuracy of the model on the validation dataset.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader),
                             [batch_time, top1, top5],
                             prefix='Test: ')

    model.eval()
    end = time.time()

    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            progress.display(i)

    logger.info('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg
