import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        print(path)
        data = torch.load(path)
        self.load_state_dict(data)
        return self.cuda()

    def save(self, name=None):
        prefix = 'snapshot/' + self.model_name + '_' +self.opt.type_+'_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix + name
        data=self.state_dict()

        torch.save(data, path)
        return path


if __name__ == "__main__":
    print(1)
    # test = Test()
    # print(test.model_name)