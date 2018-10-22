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

def train_distil_model_joint(full_model, small_model, train_iter, val_iter, full_model_args, small_model_args):
    full_model_optimizer = torch.optim.Adam(full_model.parameters(), lr = full_model_args.lr)
    small_model_optimizer = torch.optim.Adam(small_model.parameters(), lr = small_model_args.lr)
    steps = 0
    big_temperature = 10
    for epoch in range(1, full_model_args.epochs +1):
        for batch in train_iter:
            (inputs, inputs_length), target = batch.text, batch.label - 1
            # print(inputs)
            # print(target)
            target = torch.zeros(full_model_args.batch_size, full_model_args.class_num).scatter_(1, target.view(full_model_args.batch_size, 1), 1)
            full_inputs, full_inputs_length, full_target = inputs, inputs_length, target
            small_inputs, small_inputs_length, small_target = inputs, inputs_length, target
            if full_model_args.cuda and full_model_args.device != -1:
                full_inputs, full_inputs_length, full_target = inputs.cuda(), \
                                                               inputs_length.cuda(), target.cuda()

            if small_model_args.cuda and small_model_args.device != -1:
                small_inputs, small_inputs_length, small_target = inputs.cuda(), \
                                                                  inputs_length.cuda(), target.cuda()

            full_model_optimizer.zero_grad()
            small_model_optimizer.zero_grad()
            full_model_logits = full_model(full_inputs, full_inputs_length)
            small_model_logits = small_model(small_inputs, small_inputs_length)
            full_model_loss = softmax_cross_entropy_loss(self_softmax(full_model_logits, big_temperature),
                                                         full_target)
            small_model_loss1 = softmax_cross_entropy_loss(self_softmax(small_model_logits, big_temperature),
                                                           small_target)
            small_model_loss2 = softmax_cross_entropy_loss(self_softmax(small_model_logits, big_temperature),
                                                           full_model_logits)
            small_model_loss = small_model_loss1 + small_model_loss2
            full_model_loss.backward(retain_graph=True)
            small_model_loss.backward(retain_graph=True)
            full_model_optimizer.step()
            small_model_optimizer.step()
            steps += 1

            if steps % full_model_args.log_interval == 0:
                full_model_corrects = (
                            torch.max(full_model_logits, 1)[1].data == torch.max(full_target, 1)[1].data).sum()
                small_model_corrects = (
                            torch.max(small_model_logits, 1)[1].data == torch.max(small_target, 1)[1].data).sum()

                full_model_accuracy = 100 * full_model_corrects / batch.batch_size
                small_model_accuracy = 100 * small_model_corrects / batch.batch_size

                sys.stdout.write(
                    '\rBatch[{}] - full_model_loss: {:.6f}  full_model_acc: {:.4f}%({}/{})'
                    ' - small_model_loss: {:.6f}  small_model_acc: {:.4f}%({}/{})'.format(steps,
                                                                             full_model_loss.item(),
                                                                             full_model_accuracy,
                                                                             full_model_corrects,
                                                                             batch.batch_size,
                                                                             small_model_loss.item(),
                                                                             small_model_accuracy,
                                                                             small_model_corrects,
                                                                             batch.batch_size))

            if steps % full_model_args.test_interval == 0:
                dev_distil_model_joint(full_model, small_model, val_iter, full_model_args, small_model_args)


def dev_distil_model_joint(full_model, small_model, val_iter, full_model_args, small_model_args):
    full_model.eval()
    small_model.eval()
    full_model_corrects, full_model_avg_loss = 0.0, 0.0
    small_model_corrects, small_model_avg_loss = 0.0, 0.0
    small_temperature = 1
    for batch in val_iter:

        (inputs, inputs_length), target = batch.text, batch.label - 1
        target = torch.zeros(full_model_args.batch_size, full_model_args.class_num).scatter_(1, target.view(
            full_model_args.batch_size, 1), 1)

        full_inputs, full_inputs_length, full_target = inputs, inputs_length, target
        small_inputs, small_inputs_length, small_target = inputs, inputs_length, target

        if full_model_args.cuda and full_model_args.device != -1:
            full_inputs, full_inputs_length, full_target = inputs.cuda(), \
                                            inputs_length.cuda(), target.cuda()

        if small_model_args.cuda and small_model_args.device != -1:
            small_inputs, small_inputs_length, small_target = inputs.cuda(), \
                                            inputs_length.cuda(), small_model_args.cuda()


        # logit = model(inputs, inputs_length)
        # loss = F.cross_entropy(logit, target, size_average=False)
        # correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        full_model_logits = full_model(full_inputs, full_inputs_length)
        small_model_logits = small_model(small_inputs, small_inputs_length)
        full_model_loss = softmax_cross_entropy_loss(self_softmax(full_model_logits, small_temperature), full_target)
        small_model_loss1 = softmax_cross_entropy_loss(self_softmax(small_model_logits, small_temperature), small_target)
        small_model_loss2 = softmax_cross_entropy_loss(self_softmax(small_model_logits, small_temperature), full_model_logits)
        small_model_loss = small_model_loss1 + small_model_loss2
        full_model_correct = (torch.max(full_model_logits, 1)[1].data == torch.max(full_target, 1)[1].data).sum()
        small_model_correct = (torch.max(small_model_logits, 1)[1].data == torch.max(small_target, 1)[1].data).sum()

        full_model_avg_loss += full_model_loss
        full_model_corrects += full_model_correct

        small_model_avg_loss += small_model_loss
        small_model_corrects += small_model_correct

    size = len(val_iter.dataset)
    full_model_avg_loss /= size
    full_model_avg_loss /= size

    full_model_accuracy = 100.0 * full_model_corrects / size
    small_model_accuracy = 100.0 * small_model_corrects / size
    full_model.train()
    small_model.train()
    print('\nEvaluation - full_model_loss: {:.6f}  full_model_acc: {:.4f}%({}/{}) '
          'small_model_loss: {:.6f}  small_model_acc: {:.4f}%({}/{}) '.format(full_model_avg_loss,
                                                                     full_model_accuracy,
                                                                     full_model_corrects,
                                                                     size,small_model_avg_loss,
                                                                     small_model_accuracy,
                                                                     small_model_corrects,
                                                                     size))


# cross_entropy1 = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y1, 1e-10, 1.0)))
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    # args, unknown = get_common_args()
    # args.device = -1


    args_full_model, unknown = get_common_args()
    args_full_model.device = 0
    args_full_model.lstm_num_layers = 2
    args_full_model.lstm_hidden_dim = 1200

    args_small_model, unknown = get_common_args()
    args_small_model.device = 1
    args_small_model.lstm_num_layers = 2
    args_small_model.lstm_hidden_dim = 600

    print(args_full_model)
    print(args_small_model)

    train_iter, val_iter = get_dataset_iter(args_full_model, "NewsGroup")
    args_small_model.embed_num = args_full_model.embed_num
    args_small_model.class_num = args_full_model.class_num

    # -1 means cpu, else the gpu index 0,1,2,3
    # args.device = 0
    # print("args : " + str(args))
    # print("unknown args : " + str(unknown))
    #
    # model = LSTMSelfAttention(args)

    full_model = LSTMSelfAttention(args_full_model)
    small_model = LSTMSelfAttention(args_small_model)

    if args_full_model.cuda and args_full_model.device != -1:
        full_model = full_model.cuda()
    if args_small_model.cuda and args_small_model.device != -1:
        small_model = small_model.cuda()

    train_distil_model_joint(full_model, small_model, train_iter, val_iter, args_full_model, args_small_model)


    # train(model, train_iter, val_iter, args)


