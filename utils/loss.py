import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from config import opt
import torch.autograd.variable as Variable


class DiceLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100, reduce=True):
        super(DiceLoss,self).__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.softMax = nn.Softmax(dim = 1)

    def forward(self,input,target,weight=None):
        self.weight = weight
        num_classes = input.size(1)
        input = self.softMax(input)

        one_hot = target.new(num_classes,num_classes).fill_(0)

        for i in range(num_classes):
            one_hot[i,i] = 1

        target_onehot = one_hot[target]
        #print(one_hot)
        target_onehot = target_onehot.unsqueeze(1).transpose(1,-1).squeeze(-1)

        target_onehot=target_onehot.float()
        input=input.float()
        loss = self.dice_loss(input,target_onehot,self.weight)
        return loss

    def dice_loss(self,input, target, weight=None):
        smooth = 1.0
        loss = 0.0
        n_classes = input.size(1)

        for c in range(n_classes):
            iflat = input[:, c].contiguous().view(-1)
            tflat = target[:, c].contiguous().view(-1)
            intersection = (iflat * tflat).sum()

            if weight is not None:
                # must be tensor
                w = weight[c]
            else:
                w = 1 / n_classes

            w = torch.tensor(w)
            w = w.float().cuda()

            loss += w * (1 - ((2. * intersection + smooth) /
                              (iflat.sum() + tflat.sum() + smooth)))
        return loss




class MyCrossEntrophy(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100, reduce=True):
        super(MyCrossEntrophy,self).__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self,input,target,weight=None):
        num_classes = input.size(1)

        if weight is not None:
            # must be tensor
            w = torch.tensor(weight)
            w = w.float().cuda()
            self.crossEntrophy = nn.CrossEntropyLoss(w)

        else:
            self.crossEntrophy = nn.CrossEntropyLoss()

        loss =self.crossEntrophy(input,target)

        return loss



#################################################################################
class DiceLossPlusCrossEntrophy(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100, reduce=True):
        super(DiceLossPlusCrossEntrophy,self).__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.softMax = nn.Softmax(dim = 1)

    def forward(self,input,target,weight=None):
        num_classes = input.size(1)

        if weight is not None:
            # must be tensor
            w = torch.tensor(weight)
            w = w.float().cuda()
            self.crossEntrophy = nn.CrossEntropyLoss(w)

        else:
            self.crossEntrophy = nn.CrossEntropyLoss()

        loss1 = 0.5*self.crossEntrophy(input,target)

        input = self.softMax(input)
        one_hot = target.new(num_classes,num_classes).fill_(0)

        for i in range(num_classes):
            one_hot[i,i] = 1

        target_onehot = one_hot[target]
        target_onehot = target_onehot.unsqueeze(1).transpose(1,-1).squeeze(-1)

        target_onehot=target_onehot.float()
        input=input.float()
        loss2=self.dice_loss(input,target_onehot,weight)

        # the format of the cross entropy loss and dice loss should have comparable size
        # loss += self.dice_loss(input,target_onehot,weight)
        loss = loss1+loss2

        print('cross entropy loss:{} and dice loss:{}'.format(loss1,loss2))

        return loss


    def dice_loss(self,input, target, weight=None):
        smooth = 1.0
        loss = 0.0
        n_classes = input.size(1)

        for c in range(n_classes):
            iflat = input[:, c].contiguous().view(-1)
            tflat = target[:, c].contiguous().view(-1)
            intersection = (iflat * tflat).sum()

            if weight is not None:
                # must be tensor
                w = weight[c]
            else:
                w = 1 / n_classes

            w = torch.tensor(w)
            w = w.float().cuda()

            loss += w * (1 - ((2. * intersection + smooth) /
                              (iflat.sum() + tflat.sum() + smooth)))
        return loss





def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    mm1,mm2,mm3=input_logits.size()[2:]
    batch_size=input_logits.size()[0]
    whole_size = mm1 * mm2 * mm3*batch_size
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes/whole_size


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes



#https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
#https://github.com/unsky/focal-loss
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logits, targets, class_weights=None, type='softmax'):
        targets = targets.view(-1, 1).long()

        if type=='sigmoid':
            if class_weights is None: class_weights =[0.5, 0.5]

            probs  = F.sigmoid(logits)
            probs  = probs.view(-1, 1)
            probs  = torch.cat((1-probs, probs), 1)
            selects = torch.FloatTensor(len(probs), 2).zero_().cuda()
            selects.scatter_(1, targets, 1.)

        elif  type=='softmax':
            B,C,H,W = logits.size()
            if class_weights is None: class_weights =[1/C]*C

            logits  = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
            probs   = F.softmax(logits,1)
            selects = torch.FloatTensor(len(probs), C).zero_().cuda()
            selects.scatter_(1, targets, 1.)

        class_weights = torch.FloatTensor(class_weights).cuda().view(-1,1)
        weights = torch.gather(class_weights, 0, targets)


        probs      = (probs*selects).sum(1).view(-1,1)
        batch_loss = -weights*(torch.pow((1-probs), self.gamma))*probs.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weight1=None):
        if weight1 is not None:
            weight1= torch.tensor(weight1)
            weight1 = weight1.float().cuda()
            # bce = 250 *F.binary_cross_entropy_with_logits(input, target, weight1)
            bce = 250 * F.binary_cross_entropy_with_logits(input, target, weight1)
        else:
            # bce = 250 *F.binary_cross_entropy_with_logits(input, target)
            bce = 250 * F.binary_cross_entropy_with_logits(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        print('bce loss is {:.4f}, dice loss is {:.4f}'.format(bce,dice))
        return bce + dice

