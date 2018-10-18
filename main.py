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

def train_distil_model(full_model, small_model, train_iter, val_iter, full_model_args, small_model_args):
    full_model_optimizer = torch.optim.Adam(full_model.parameters(), lr = full_model_args.lr)
    small_model_optimizer = torch.optim.Adam(small_model.parameters(), lr = small_model_args.lr)



# cross_entropy1 = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y1, 1e-10, 1.0)))
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    # args, unknown = get_common_args()
    # args.device = -1


    args_full_model, unknown = get_common_args()
    args_full_model.device = -1
    args_full_model.lstm_num_layers = 2
    args_full_model.lstm_hidden_dim = 1200

    args_small_model, unknown = get_common_args()
    args_small_model.device = -1
    args_small_model.lstm_num_layers = 2
    args_small_model.lstm_hidden_dim = 600

    print(args_full_model)
    print(args_small_model)

    full_model = LSTMSelfAttention(args_full_model)
    small_model = LSTMSelfAttention(args_small_model)

    # -1 means cpu, else the gpu index 0,1,2,3
    # args.device = 0
    # print("args : " + str(args))
    # print("unknown args : " + str(unknown))
    # train_iter, val_iter = get_dataset_iter(args, "NewsGroup")
    #
    # model = LSTMSelfAttention(args)
    # if args.cuda and args.device != -1:
    #     torch.cuda.set_device(args.device)
    #     model = model.cuda()
    #
    # train(model, train_iter, val_iter, args)


