from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

# Took from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
def repackage_state(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_state(v) for v in h)


class Seq2Seq(nn.Module):
    def __init__(self, encode_ntoken, decode_ntoken,
            input_size, hidden_size,
            input_max_len, output_max_len,
            batch_size,
            nlayers=1, bias=False, attention=True, dropout_p=0.5,
            batch_first=False):
        super(Seq2Seq, self).__init__()

        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

        # encoder stack
        self.enc_embedding = nn.Embedding(encode_ntoken, input_size)
        self.encoder = nn.LSTM(input_size, hidden_size, nlayers, bias=bias, batch_first=batch_first)

        # decoder stack
        self.dec_embedding = nn.Embedding(decode_ntoken, input_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, nlayers, bias=bias, batch_first=batch_first)
        if attention:
            self.attn_enc_linear = nn.Linear(hidden_size, hidden_size)
            self.attn_dec_linear = nn.Linear(hidden_size, hidden_size)
            self.attn_tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_size, decode_ntoken, bias=True)
        self.softmax = nn.LogSoftmax()

        self.decode_ntoken = decode_ntoken
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.attention = attention
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len
        self.batch_first = batch_first

    def init_weights(self, initrange):
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def attention_func(self, encoder_outputs, decoder_hidden):
        # (batch_size*input_len*hidden_len) * (batch_size*hidden_len) -> (batch_size*input_len)
        dot = torch.bmm(torch.transpose(encoder_outputs, 0, 1), decoder_hidden.unsqueeze(2))
        # -> (batch_size*input_len)
        attention = nn.Softmax()(dot.squeeze())
        # (batch_size*input_len*hidden_len)' * (batch_size*input_len) -> (batch_size*hidden_len)
        enc_attention = torch.bmm(torch.transpose(torch.transpose(encoder_outputs, 0, 1), 1, 2), attention.unsqueeze(2))
        # (batch_size*hidden_len)*(hidden_len*hidden_len) + (batch_size*hidden_len)*(hidden_len*hidden_len)
        # -> (batch_size*hidden_len)
        hidden = self.attn_tanh(self.attn_enc_linear(enc_attention.squeeze()) + self.attn_dec_linear(decoder_hidden))
        return hidden

    def encode(self, encoder_inputs):
        weight = next(self.parameters()).data
        init_state = (Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_()))
        embedding = self.enc_embedding(encoder_inputs)
        if self.dropout_p > 0:
            embedding = self.dropout(embedding)
        encoder_outputs, encoder_state = self.encoder(embedding, init_state)
        return encoder_outputs, encoder_state

    def decode(self, encoder_outputs, encoder_state, decoder_inputs, feed_previous):
        pred = []
        state = encoder_state
        if feed_previous:
            if self.batch_first:
                embedding = self.dec_embedding(decoder_inputs[:,0].unsqueeze(1))
            else:
                embedding = self.dec_embedding(decoder_inputs[0].unsqueeze(0))
            for time in range(1, self.output_max_len):
                state = repackage_state(state)
                # batch_size * 1 * embedding_size
                output, state = self.decoder(embedding, state)
                # print(output.size())

                softmax = self.predict(output.squeeze(), encoder_outputs)
                # feed previous
                decoder_input = softmax.max(1)[1]
                embedding = self.dec_embedding(decoder_input.squeeze().unsqueeze(0))
                if self.batch_first:
                    embedding = torch.transpose(embedding, 0, 1)
                pred.append(softmax)
        else:
            embedding = self.dec_embedding(decoder_inputs)
            if self.dropout_p > 0:
                embedding = self.dropout(embedding)
            outputs, _ = self.decoder(embedding, state)
            # print(outputs.size())

            # if self.batch_first:
                # for batch in range(self.batch_size):
                    # output = outputs[batch,1:,:]
                    # softmax = self.predict(output, encoder_outputs)
                    # pred.append(softmax)
            # else:
            for time_step in range(self.output_max_len - 1):
                if self.batch_first:
                    output = outputs[:,time_step,:]
                else:
                    output = outputs[time_step]
                softmax = self.predict(output, encoder_outputs)
                pred.append(softmax)
        return pred

    def predict(self, dec_output, encoder_outputs):
        if self.attention:
            hidden = self.attention_func(encoder_outputs, dec_output)
        else:
            hidden = dec_output
        if self.dropout_p > 0:
            hidden = self.dropout(hidden)
        linear = self.linear(hidden)
        softmax = self.softmax(linear)
        return softmax

    def forward(self, encoder_inputs, decoder_inputs, feed_previous=False):
        # encoding
        encoder_outputs, encoder_state = self.encode(encoder_inputs)

        # decoding
        pred = self.decode(encoder_outputs, encoder_state, decoder_inputs, feed_previous)
        return pred

class Seq2Tree(Seq2Seq):
    def decode(self, encoder_outputs, encoder_state, decoder_inputs, feed_previous):
        pass

    def forward(self, encoder_inputs, decoder_tree, feed_previous=False):
        # encoding
        encoder_outputs, encoder_state = self.encode(encoder_inputs)

        # decoding
        pred = self.decode(encoder_outputs, encoder_state, decoder_tree, feed_previous)
        return pred
