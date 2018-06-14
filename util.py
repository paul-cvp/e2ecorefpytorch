import os
import errno
import collections
import json

import numpy as np
# import tensorflow as tf
import torch
import torch as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pyhocon


def make_summary(value_dict):
    # return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])
    summary = [(k, v) for k, v in value_dict.items()]
    return summary


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_config(filename):
    return pyhocon.ConfigFactory.parse_file(filename)


def print_config(config):
    print(pyhocon.HOCONConverter.convert(config, "hocon"))


def set_gpus(*gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with open(char_vocab_path) as f:
        for c in f.readlines():
            vocab.extend(c.strip())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def load_embedding_dict(embedding_path, embedding_size, embedding_format):
    print("Loading word embeddings from {}...".format(embedding_path))
    default_embedding = np.zeros(embedding_size)
    embedding_dict = collections.defaultdict(lambda: default_embedding)
    skip_first = embedding_format == "vec"
    with open(embedding_path) as f:
        for i, line in enumerate(f.readlines()):
            if skip_first and i == 0:
                continue
            splits = line.split()
            assert len(splits) == embedding_size + 1
            word = splits[0]
            embedding = np.array([float(s) for s in splits[1:]])
            embedding_dict[word] = embedding
    print("Done loading word embeddings.")
    return embedding_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=None):
    # TODO figure out the reshaping
    projected_layer = nn.Linear(inputs.shape[1], output_size)
    if initializer is not None:
        projected_layer.apply(initializer)
    return projected_layer(inputs)
    # return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def shape(x, dim):
    return x.size()[dim] or x.shape[dim]


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
    if len(inputs.get_shape()) > 2:
        current_inputs = inputs.view([-1, shape(inputs, -1)])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.zeros([shape(current_inputs, 1), hidden_size])
        nn.init.xavier_uniform(hidden_weights)
        hidden_bias = tf.zeros([hidden_size])
        nn.init.constant(hidden_bias)
        current_outputs = F.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

        if dropout is not None:
            current_outputs = F.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.zeros([shape(current_inputs, 1), output_size])
    output_weights.apply(output_weights_initializer)
    output_bias = tf.zeros([output_size])
    nn.init.constant(output_bias)
    outputs = tf.matmul(current_inputs, output_weights) + output_bias

    if len(inputs.get_shape()) == 3:
        outputs = outputs.view([shape(inputs, 0), shape(inputs, 1), output_size])
    elif len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
    return outputs


def cnn(inputs, filter_sizes, num_filters):
    num_words = shape(inputs, 0)  # all words from all sentences
    num_chars = shape(inputs, 1)  # longest word in characters
    input_size = shape(inputs, 2)  # feature embeddings 8
    outputs = []
    for i, filter_size in enumerate(filter_sizes):  # [3,4,5]
        # with tf.variable_scope("conv_{}".format(i)):
        w = tf.randn([filter_size, input_size, num_filters])  # [3,8,50][4,8,50][5,8,50]
        w = nn.init.xavier_uniform(w)
        b = tf.randn([num_filters])  # [50]x3
        conv = F.conv1d(inputs, Variable(w), stride=1, padding=0,
                        bias=Variable(b))  # [num_words, num_chars - filter_size, num_filters]
        # [num_words, num_chars - [3,4,5], 50]
        h = F.relu(conv)  # [num_words, num_chars - filter_size, num_filters]
        pooled = tf.max(h, 1)  # [num_words, num_filters]
        outputs.append(pooled)
    return tf.cat(outputs, 1)  # [num_words, num_filters * len(filter_sizes)] [num_words, 50*3]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1


class CustomLSTMCell(nn.RNNCell):
    def __init__(self, num_units, batch_size, dropout):
        super(CustomLSTMCell, self).__init__()
        self._num_units = num_units
        self._dropout = dropout
        self._dropout_mask = F.dropout(tf.ones([batch_size, self.output_size]), dropout)
        self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
        initial_cell_state = tf.empty([1, self.output_size])
        nn.init.xavier_uniform(initial_cell_state)
        initial_hidden_state = tf.empty([1, self.output_size])
        nn.init.xavier_uniform(initial_hidden_state)
        # self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)
        self._initial_state = nn.LSTMCell(initial_cell_state, initial_hidden_state)

    @property
    def state_size(self):
        return nn.LSTMCell(self.output_size, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def initial_state(self):
        return self._initial_state

    def preprocess_input(self, inputs):
        # projection is only initializing a Linear Layer in this case
        return projection(inputs, 3 * self.output_size)

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        # with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
        c, h = state
        h *= self._dropout_mask
        projected_h = projection(h, 3 * self.output_size, initializer=self._initializer)
        concat = inputs + projected_h
        i, j, o = tf.split(concat, split_size=3, dim=1)
        i = F.sigmoid(i)
        new_c = (1 - i) * c + i * F.tanh(j)
        new_h = F.tanh(new_c) * F.sigmoid(o)
        new_state = nn.LSTMCell(new_c, new_h)
        return new_h, new_state

    def _orthonormal_initializer(self, scale=1.0):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
            M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return params

        return _initializer

    def _block_orthonormal_initializer(self, output_sizes):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            assert len(shape) == 2
            assert sum(output_sizes) == shape[1]
            initializer = self._orthonormal_initializer()
            params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
            return params

        return _initializer


def get_range_vector(size: int, device: int) -> torch.Tensor:
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(indices: torch.Tensor,
                                    sequence_length: int) -> torch.Tensor:
    # Shape: (batch_size)
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: torch.LongTensor = None) -> torch.Tensor:
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def flattened_index_select(target: torch.Tensor,
                           indices: torch.LongTensor) -> torch.Tensor:
    if indices.dim() != 2:
        raise ValueError("Indices passed to flattened_index_select had shape {} but "
                         "only 2 dimensional inputs are supported.".format(indices.size()))
    # Shape: (batch_size, set_size * subset_size, embedding_size)
    flattened_selected = target.index_select(1, indices.view(-1))

    # Shape: (batch_size, set_size, subset_size, embedding_size)
    selected = flattened_selected.view(target.size(0), indices.size(0), indices.size(1), -1)
    return selected
