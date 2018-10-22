#coding=utf8
from utils import *
from models.LSTMSelfAttention import *
from mydatasets import *
import sys

def validate(model, val_iter, args):
    model.eval()
    corrects, avg_loss = 0.0, 0.0
    for batch in val_iter:

        (inputs, inputs_length), target = batch.text, batch.label - 1

        if args.cuda and args.device != -1:
            inputs, inputs_length, target = inputs.cuda(), inputs_length.cuda(), target.cuda()

        logit = model(inputs, inputs_length)
        loss = F.cross_entropy(logit, target, size_average=False)
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        avg_loss += loss.item()
        corrects += correct

    size = len(val_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
def train(model, train_iter, val_iter, args):
    print("begin to train models...")
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    steps = 0

    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            (inputs, inputs_length), target = batch.text, batch.label - 1
            if args.cuda and args.device != -1:
                inputs, inputs_length, target = inputs.cuda(), inputs_length.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(inputs, inputs_length)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

                accuracy = 100*corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))

            if steps % args.test_interval == 0:
                validate(model, val_iter, args)


if __name__ == "__main__":
    args_small_model, unknown = get_common_args()
    args_small_model.device = 1
    args_small_model.lstm_num_layers = 2
    args_small_model.lstm_hidden_dim = 600
    print(args_small_model)
    train_iter, val_iter = get_dataset_iter(args_small_model, "NewsGroup")
    small_model = LSTMSelfAttention(args_small_model)
    if args_small_model.cuda and args_small_model.device != -1:
        small_model = small_model.cuda()
    train(small_model, train_iter, val_iter, args_small_model)
