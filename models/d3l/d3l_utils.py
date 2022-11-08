import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value

class MSE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return (F.mse_loss(x.softmax(1), y.softmax(1).detach(), reduction='none').mean(1))
