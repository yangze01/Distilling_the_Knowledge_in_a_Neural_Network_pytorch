import torch
batch_size = 5
nb_digits = 10

logits = [[0.3, 0.2, 0.5],
     [0.7, 0.1, 0.2]]

target = [[0, 1, 0],
          [1, 0, 0]]
logits = torch.FloatTensor(logits)
target = torch.FloatTensor(target)


# print(torch.mean(logits * target))

def softmax_cross_entropy_loss(outputs, labels):
    log_value = torch.log(outputs)
    clamp_value = log_value#lamp(log_value, 0.0, 1.0)
    cross_entropy = torch.mean(labels * clamp_value)
    return cross_entropy

def self_softmax(logits, temperature =1):
    softmax_logits = torch.softmax(logits/float(temperature), dim = 1)
    return softmax_logits

print(softmax_cross_entropy_loss(self_softmax(logits), target))