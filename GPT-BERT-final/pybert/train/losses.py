from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import BCELoss


__call__ = ['CrossEntropy','BCEWithLogLoss','BCELoss']

class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss

class MultiLabelCrossEntropy(object):
    def __init__(self):
        pass
    def __call__(self, output, target):
        loss = CrossEntropyLoss(reduction='none')(output,target)
        return loss

class BCEWithLoss(object):
    def __init__(self):
        self.loss_fn = BCELoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss


