import time

import numpy as np
import torch
from torch import nn
from torch import optim

from config import device, label_names, print_every, hidden_size, encoder_n_layers, dropout, learning_rate, \
    epochs
from data_gen import SaDataset
from models import EncoderRNN
from utils import AverageMeter, ExpoAverageMeter, accuracy, Lang, timestamp, adjust_learning_rate, save_checkpoint


def train(epoch, train_data, encoder, optimizer):
    '''
    训练
    :param epoch:
    :param train_data:
    :param encoder:
    :param optimizer: 优化器算法 sgd -> adam ->adagrad -> rmsprop 看torch文档
    :return:
    '''
    # Ensure dropout layers are in train mode
    # 将模型状态设置为训练状态（会启用某些机制  ps：模型状态一般有 train 和 eval
    encoder.train()

    # Loss function
    # 模型（评判函数），没有状态  loop:  input -> model ->output-criterion->loss  loss影响model参数，我们期望loss下降
    criterion = nn.CrossEntropyLoss().to(device)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    accs = ExpoAverageMeter()  # accuracy

    start = time.time()

    # Batches
    # 开始训练，监督学习
    for i_batch, (input_variable, lengths, target_variable) in enumerate(train_data):
        # 每个tensor包含输入的数据和grad（梯度），
        # Zero gradients 清零所有tensor的grad
        optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        # print('input_variable.size(): ' + str(input_variable.size()))
        # print('lengths.size(): ' + str(lengths.size()))
        # print('target_variable.size(): ' + str(target_variable.size()))

        # Forward pass through encoder
        outputs = encoder(input_variable, lengths)
        # print('outputs.size(): ' + str(outputs.size()))

        loss = 0
        acc = 0

        # 使用criterion函数对结果进行评价
        for idx, _ in enumerate(label_names):
            loss += criterion(outputs[:, :, idx], target_variable[:, idx]) / len(label_names)
            acc += accuracy(outputs[:, :, idx], target_variable[:, idx]) / len(label_names)

        # 根据loss算所有参数的梯度，并给tensor.grad赋值
        loss.backward()

        # 开始优化被允许优化的参数
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)
        accs.update(acc)

        start = time.time()

        # Print status
        if i_batch % print_every == 0:
            print('[{0}] Epoch: [{1}][{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(timestamp(), epoch, i_batch, len(train_data),
                                                                    batch_time=batch_time,
                                                                    loss=losses,
                                                                    accs=accs))


def valid(val_data, encoder):
    '''
    跑认证数据，避免过拟合，（意思是避免模型只在训练的数据上运行结果很好
    :param val_data:
    :param encoder:
    :return:
    accs.avg, losses.avg
    '''
    encoder.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    # todo
    criterion = nn.CrossEntropyLoss().to(device)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)
    accs = AverageMeter()  # accuracy

    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, (input_variable, lengths, target_variable) in enumerate(val_data):
            # Set device options
            input_variable = input_variable.to(device)
            lengths = lengths.to(device)
            target_variable = target_variable.to(device)

            outputs = encoder(input_variable, lengths)

            loss = 0
            acc = 0

            for idx, _ in enumerate(label_names):
                loss += criterion(outputs[:, :, idx], target_variable[:, idx]) / len(label_names)
                acc += accuracy(outputs[:, :, idx], target_variable[:, idx]) / len(label_names)

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            accs.update(acc)

            start = time.time()

            # Print status
            if i_batch % print_every == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(i_batch, len(val_data),
                                                                        batch_time=batch_time,
                                                                        loss=losses,
                                                                        accs=accs))

    return accs.avg, losses.avg


def main():
    # 加载词库
    voc = Lang('data/WORDMAP.json')
    print("voc.n_words: " + str(voc.n_words))
    train_data = SaDataset('train', voc)
    val_data = SaDataset('valid', voc)

    # Initialize encoder
    # todo
    '''
    词表里面的单词数量
    隐含层大小
    rnn层数
    某个机制...
    '''
    encoder = EncoderRNN(voc.n_words, hidden_size, encoder_n_layers, dropout)

    # Use appropriate device ,此时模型都是初始值
    encoder = encoder.to(device)

    # Initialize optimizers  创建优化器，梯度下降，能自动调整模型的参数
    # optim是一个pytorch的一个包，adam是一个优化算法，梯度下降
    print('Building optimizers ...')
    # todo
    # 模型所有参数，lr学习率
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    best_acc = 0
    # todo
    epochs_since_improvement = 0

    # Epochs 所有数据都跑一遍，叫做epochs
    for epoch in range(0, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(epoch, train_data, encoder, optimizer)

        # One epoch's validation
        val_acc, val_loss = valid(val_data, encoder)
        print('\n * ACCURACY - {acc:.3f}, LOSS - {loss:.3f}\n'.format(acc=val_acc, loss=val_loss))

        # Check if there was an improvement
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, encoder, optimizer, val_acc, is_best)

        # Reshuffle samples
        np.random.shuffle(train_data.samples)
        np.random.shuffle(val_data.samples)


if __name__ == '__main__':
    main()
