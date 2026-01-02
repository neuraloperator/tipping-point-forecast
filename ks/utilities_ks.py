### Adapted from official neural operator repository: https://github.com/neuraloperator/neuraloperator

import torch
import operator
from functools import reduce

#################################################
#
# Utilities:
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y, std):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if std == True:
            return torch.std(diff_norms / y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms


    def __call__(self, x, y, std=False):
        return self.rel(x, y, std)

# Weighted relative LpLoss across one dimension
class WeightedLpLoss(object):
    """
        Computes a weighted Lp loss between two tensors. In particular, the weighting
        is applied across a single `axis`. For instance, if the data is of the shape
        (5, 4, 3, 2), and if we want to weight across axis 1, then the data is 
        reshaped to (5, 4, 6), and the norm is taken across the final dimension.
        
        The mean is then computed across all dimensions from 1 to `axis`.
    """
    def __init__(self, sample_weights, axis, p=2, size_average=True, reduction=True):
        super(WeightedLpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert p > 0 and axis >= 0 and len(sample_weights.shape) == 1

        self.sample_weights = sample_weights
        self.p = p
        self.reduction = reduction
        self.axis = axis # axis to weight over
        self.size_average = size_average

    def __call__(self, x, y):
        if x.shape != y.shape:
            raise RuntimeError("x and y must have the same shape")
        elif len(x.shape) <= self.axis:
            raise RuntimeError("Inputs must have at least {} dimensions".format(self.axis))
        elif self.axis == len(x.shape):
            raise RuntimeError("Axis to weight over cannot be last axis.")
        elif len(self.sample_weights) != x.shape[self.axis]:
            raise RuntimeError("Sample weights and inputs must agree in size across dimension {}.".format(self.axis))

        # Reshape x and y accordingly
        new_shape = x.shape[:self.axis + 1] + (-1,)
        x = x.reshape(new_shape)
        y = y.reshape(new_shape)

        sample_weights = self.sample_weights
        for i in range(self.axis):
            sample_weights = sample_weights.unsqueeze(0)

        num_examples = x.size()[0]

        squared_norms = torch.square(torch.norm(x - y, self.p, [i for i in range(self.axis + 1, len(x.shape))]))
        sample_weights = sample_weights.repeat(squared_norms.shape[:-1] + (1,)) # shape sample_weights appropriately

        weighted_norms = torch.sqrt(sample_weights * squared_norms)
        diff_norms = torch.mean(weighted_norms, [i for i in range(1, self.axis + 1)])

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms)
            else:
                return torch.sum(diff_norms)
        return diff_norms


class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c