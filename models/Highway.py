from models.BasicModule import *
import torch.nn.functional as F


class Highway(BasicModule):
    def __init__(self, input_dim, activate_funcation = F.relu):
        super(Highway, self).__init__()
        self.nonlinear = nn.Linear(input_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.activate_function = activate_funcation

    def forward(self, x):
        """
        :param x: (B, H)
        :return: (B, H)
        """
        # (B, H) -> (B, H)
        T = F.sigmoid(self.gate(x))
        # (B, H) -> (B, H)
        H = self.activate_function(self.nonlinear(x))
        # output = T(x, W_T) * H(x, W_H) + (1 - T(x, W_t)) * x
        output = T * H + (1 - T) * x
        return output
