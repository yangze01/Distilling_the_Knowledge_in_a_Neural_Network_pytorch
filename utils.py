import argparse
import torch
import torch.nn.functional as F

#cross_entropy1 = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y1, 1e-10, 1.0)))
#temp = F.sigmoid(outputs)
#res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
#res = torch.mean(torch.sum(res, dim=1))
#t.clamp(a, min=2, max=4)

# def softmax_cross_entropy_loss(outputs, labels):
#     cross_entropy = torch.mean(labels * torch.clamp(torch.log(outputs), 0, 1))
#     return cross_entropy

def softmax_cross_entropy_loss(outputs, labels):
    log_value = torch.log(outputs)
    clamp_value = log_value#lamp(log_value, 0.0, 1.0)
    cross_entropy = -torch.mean(labels * clamp_value)
    return cross_entropy


def self_softmax(logits, temperature =1):
    softmax_logits = torch.softmax(logits/float(temperature), dim = 1)
    return softmax_logits

def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True, help='enable the cuda', required=False)
    parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')

    # models args
    parser.add_argument("-static", type=bool, default=False, help="fix the embedding [default: False]")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=256, help='Number of training epochs', required=False)
    parser.add_argument("-add_pretrain_embedding", type=bool, default=False)

    # LSTM args
    parser.add_argument('--embed_dim', type=int, default=128, help='the embedding dim of word embedding', required=False)
    parser.add_argument("--lstm_num_layers", type=int, default=1, help='the number of layer')
    parser.add_argument('--lstm_hidden_dim', type=int, default=100, help='Number of lstm hidden dim', required=False)
    parser.add_argument('--dropout_rate', type=int, default=0.5, help='the dropout rate', required=False)

    args, unknown = parser.parse_known_args()
    return args, unknown