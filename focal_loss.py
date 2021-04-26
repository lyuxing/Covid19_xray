import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from resnet_3d_cbam import *


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


if __name__ == "__main__":
    num_classes = 6
    input_tensor = torch.autograd.Variable(torch.rand(12, 4, 24, 256, 256)).cuda()
    model = resnet10(sample_size=256, sample_duration=24, num_classes=num_classes).cuda()
    # model = ResNet(dataset = 'calc', depth = 34, num_classes=num_classes).cuda()
    output = model(input_tensor)
    output = torch.sigmoid(output)

    label = torch.autograd.Variable(torch.tensor(np.array([1,1,0,0,1,0]*12)))
    target = label.float().reshape(-1, num_classes).cuda()

    criterion = nn.BCELoss().cuda()

    loss = criterion(output,target)

    w = torch.tensor([4,4,2,2,1,1]).float().cuda()
    Floss = FocalLoss(reduce=False).forward(inputs= output,targets=target)
    wFloss = Floss*w
    Floss  = FocalLoss().forward(inputs= output,targets=target)
    wFloss = torch.mean(wFloss)

    print(loss,Floss,wFloss)

