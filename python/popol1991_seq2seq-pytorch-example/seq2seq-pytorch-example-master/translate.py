from __future__ import print_function

import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from six.moves import xrange

import data_utils
from seq2seq import Seq2Seq, Seq2Tree

criterion = nn.NLLLoss()

def get_batch(data, encoder_size, decoder_size, batch_size, reverse_input=False, batch_first=False):
    encoder_inputs = []
    decoder_inputs = []
    for _ in xrange(batch_size):
        encoder_input, decoder_input = random.choice(data)
        encoder_input += [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(encoder_input)
        decoder_input = [data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID] * (decoder_size - len(decoder_input) - 1)
        decoder_inputs.append(decoder_input)

    if batch_first:
        batch_encoder_inputs = encoder_inputs
        batch_decoder_inputs = decoder_inputs
    else:
        batch_encoder_inputs = []
        for time in xrange(encoder_size):
            seq = [encoder_inputs[batch_idx][time] for batch_idx in xrange(batch_size)]
            if reverse_input:
                seq.reverse()
            batch_encoder_inputs.append(seq)

        batch_decoder_inputs = []
        for time in xrange(decoder_size):
            seq = [decoder_inputs[batch_idx][time] for batch_idx in xrange(batch_size)]
            batch_decoder_inputs.append(seq)

    batch_weight = torch.ones(batch_size)

    return Variable(torch.LongTensor(batch_encoder_inputs)), Variable(torch.LongTensor(batch_decoder_inputs)), Variable(batch_weight)

def evaluate(model, dev, encode_max_len, decode_max_len, batch_size, batch_first=False):
    total_loss = 0
    for _ in xrange(int(len(dev) / batch_size)):
        encoder_inputs, decoder_inputs, _ = get_batch(dev, encode_max_len, decode_max_len, batch_size, batch_first=batch_first)
        if torch.cuda.is_available():
            encoder_inputs, decoder_inputs = encoder_inputs.cuda(), decoder_inputs.cuda()
        pred = model(encoder_inputs, decoder_inputs, feed_previous=True)
        for time in xrange(len(decoder_inputs) - 1):
            y_pred = pred[time]
            if batch_first:
                target = decoder_inputs[:, time+1]
            else:
                target = decoder_inputs[time + 1]
            # print("Prediction")
            # for batch in range(batch_size):
                # print(dec_dict[y_pred[batch].max(0)[1].data[0]], end=" ")
            # print()
            # print("Target")
            # for batch in range(batch_size):
                # print(dec_dict[target[batch].data[0]], end=" ")
            # print()
            loss = criterion(y_pred, target)
            total_loss += loss.data
    return total_loss[0] / len(dev)

if __name__ == '__main__':
    model = "seq2seq"
    batch_first = True
    attention = False
    dropout_p = 0
    batch_size = 20
    encode_ntoken = 250
    decode_ntoken = 62
    encode_max_len = 20
    decode_max_len = 30
    embedding_size = 16
    hidden_size = 16
    init_range = 0.08
    step_per_epoch = 50
    learning_rate = 0.01
    learning_rate_decay = 0.98
    decay_rate = 0.95
    momentum=0.9
    data_path = "./tmp"

    checkpoint_after = 50

    train, dev, test = data_utils.prepare_data(data_path, encode_ntoken, decode_ntoken, recreate=False, model=model)
    enc_dict = dict()
    for idx, line in enumerate(open(os.path.join(data_path, "vocab.q")).readlines()):
        enc_dict[idx] = line.strip()
    dec_dict = dict()
    for idx, line in enumerate(open(os.path.join(data_path, "vocab.lf")).readlines()):
        dec_dict[idx] = line.strip()

    if model == "seq2seq":
        model = Seq2Seq(encode_ntoken, decode_ntoken,
                embedding_size, hidden_size,
                encode_max_len, decode_max_len,
                batch_size, attention=attention, dropout_p=dropout_p, batch_first=batch_first)
    else:
        model = Seq2Tree(encode_ntoken, decode_ntoken,
                embedding_size, hidden_size,
                encode_max_len, decode_max_len,
                batch_size, attention=attention, dropout_p=dropout_p, batch_first=batch_first)

    model_path = "model.dat"
    if os.path.exists(model_path):
        saved_state = torch.load(model_path)
        model.load_state_dict(saved_state)
    else:
        if torch.cuda.is_available():
            model.cuda()
        model.init_weights(init_range)
        optimizer = optim.RMSprop(model.parameters(), lr = learning_rate, alpha=decay_rate)

        train_loss = 0
        last_train_loss = 10
        loss_count = 0
        best_dev_loss = 10
        step = 0
        begin_time = time.time()
        while True:
            batch_enc_inputs, batch_dec_inputs, batch_weights = get_batch(train, encode_max_len, decode_max_len, batch_size, batch_first=batch_first)
            if torch.cuda.is_available():
                batch_enc_inputs, batch_dec_inputs = batch_enc_inputs.cuda(), batch_dec_inputs.cuda()
            pred = model(batch_enc_inputs, batch_dec_inputs, feed_previous=False)
            total_loss = None
            # if batch_first:
                # for batch in xrange(batch_size):
                    # y_pred = pred[batch]
                    # target = batch_dec_inputs[batch][1:]
                    # loss = criterion(y_pred, target)
                    # if total_loss is None:
                        # total_loss = loss
                    # else:
                        # total_loss += loss
            # else:
            for time_step in xrange(len(batch_dec_inputs) - 1):
                y_pred = pred[time_step]
                if batch_first:
                    target = batch_dec_inputs[:, time_step + 1]
                else:
                    target = batch_dec_inputs[time_step+1]
                # print(y_pred.size(), target.size())
                loss = criterion(y_pred, target)
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            optimizer.zero_grad()
            total_loss /= batch_size
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss

            if step % step_per_epoch == 0:
                epoch = step / step_per_epoch
                dev_loss = evaluate(model, dev, encode_max_len, decode_max_len, batch_size, batch_first=batch_first)
                train_loss = train_loss.data[0] / step_per_epoch
                if train_loss > last_train_loss:
                    loss_count += 1
                    if loss_count == 3:
                        learning_rate *= learning_rate_decay
                        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=decay_rate)
                        loss_count = 0
                else:
                    loss_count = 0
                last_train_loss = train_loss
                if epoch > 0:
                    epoch_time = (time.time() - begin_time) / epoch
                else:
                    epoch_time = 0
                print("Epoch time: {0}\tEpoch: {1}\tLR: {2}, Train loss: {3}\tDev loss: {4}".format(
                    epoch_time, epoch, learning_rate, train_loss, dev_loss
                    ))
                train_loss = 0
                if epoch > checkpoint_after and dev_loss < best_dev_loss:
                    state_to_save = model.state_dict()
                    torch.save(state_to_save, model_path)

            step += 1

    while True:
        enc, dec, _ = get_batch(test, encode_max_len, decode_max_len, batch_size, batch_first=batch_first)
        print("Input: ")
        for time in xrange(encode_max_len):
            if batch_first:
                print(enc_dict[enc[0][time].data[0]], end=" ")
            else:
                print(enc_dict[enc[time][0].data[0]], end=" ")

        pred = model(enc, dec, feed_previous=True)

        print("Output: ")
        for time in xrange(decode_max_len - 1):
            y_pred = pred[time][0]
            print(dec_dict[y_pred.max(0)[1].data[0]], end=" ")
        input()
