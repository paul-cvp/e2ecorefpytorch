import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.utils import log_sum_exp
from torch import optim
from torch.autograd import Variable

import util
import coref_ops
import conll
import metrics
import scipy.misc as sp


class CorefModel(nn.Module):

    def __init__(self, config):
        super(CorefModel, self).__init__()
        self.config = config
        self.config = config
        self.embedding_info = [(emb["size"], emb["lowercase"]) for emb in
                               config["embeddings"]]  # [(300,false)(50,false)]
        self.embedding_size = sum(size for size, _ in self.embedding_info)  # 350 = 300+50
        self.char_embedding_size = config["char_embedding_size"]  # 8
        self.char_dict = util.load_char_dict(config["char_vocab_path"])  # all characters + <unk> size 115
        self.max_mention_width = config["max_mention_width"]  # 10
        self.genres = {g: i for i, g in enumerate(config["genres"])}

        self.char_embeddings = nn.Parameter(torch.randn([len(self.char_dict), self.config["char_embedding_size"]]))
        self.char_cnn = CNN()
        # TODO check if the input to the BILSTM should be a pack(_padded)_sequence so that minibatches can be used
        self.bilstm = nn.LSTM(input_size=500, hidden_size=200, num_layers=1, dropout=0.2, bidirectional=True)
        self.genre_tensor = nn.Parameter(torch.randn([len(self.genres), self.config["feature_size"]]))
        self.mention_width_tensor = nn.Parameter(torch.randn([self.config["max_mention_width"], self.config["feature_size"]]))
        self.head_scores = nn.Linear(400, 1)
        self.mention = FFNNMention()
        self.same_speaker_emb = nn.Parameter(torch.randn([2, self.config["feature_size"]]))
        self.mention_distance_emb = nn.Parameter(torch.zeros([10, self.config["feature_size"]]))
        self.antecedent = FFNNAntecedent()

        nn.init.xavier_uniform_(self.char_embeddings)
        self.weights_init(self.char_cnn.parameters())
        self.hidden = self.bilstm_init(self.bilstm.hidden_size)
        nn.init.xavier_uniform_(self.genre_tensor)
        nn.init.xavier_uniform_(self.mention_width_tensor)
        self.weights_init(self.mention.parameters())
        nn.init.xavier_uniform_(self.same_speaker_emb)
        nn.init.xavier_uniform_(self.mention_distance_emb)
        self.weights_init(self.antecedent.parameters())

        # coreference score = mention score span 1 + mention score span 2 + pairwise antecedent score of both spans

    def bilstm_init(self, hidden_dim, num_layers=1):
        h_0 = torch.randn(2, num_layers, hidden_dim)
        c_0 = torch.randn(2, num_layers, hidden_dim)
        nn.init.orthogonal_(h_0)
        nn.init.orthogonal_(c_0)
        return h_0, c_0

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.xavier_uniform_(m.bias.data)

    def forward(self, word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
        training_num = 0.0
        if is_training == 1:
            training_num = 1.0

        self.dropout = 1 - (training_num * self.config["dropout_rate"])  # 0.2
        self.lexical_dropout = 1 - (training_num * self.config["lexical_dropout_rate"])  # 0.5

        num_sentences = word_emb.shape[0]  # number of sentences to predict from
        max_sentence_length = word_emb.shape[1]  # maybe caused by applying padding to the dataset to have all sentences in the same shape

        text_emb_list = [word_emb]  # 3D tensor added in an array

        if self.config["char_embedding_size"] > 0:  # true is 8
            char_emb = torch.index_select(self.char_embeddings,
                                          0,
                                          char_index.view(-1)).view(num_sentences,
                                                                    max_sentence_length,
                                                                    -1,
                                                                    self.config["char_embedding_size"])
            # [num_sentences, max_sentence_length, max_word_length, emb]
            # [a vector of embedding 8 for each character for each word for each sentence for all sentences]
            # (according to longest word and longest sentence)

            flattened_char_emb = char_emb.view([num_sentences * max_sentence_length,
                                                util.shape(char_emb, 2),
                                                util.shape(char_emb, 3)])
            # [num_sentences * max_sentence_length, max_word_length, emb]

            flattened_aggregated_char_emb = self.char_cnn(flattened_char_emb)

            # [num_sentences * max_sentence_length, emb] character level CNN

            aggregated_char_emb = flattened_aggregated_char_emb.view([num_sentences, max_sentence_length,
                                                                      util.shape(flattened_aggregated_char_emb,
                                                                                 1)])
            # [num_sentences, max_sentence_length, emb]
            text_emb_list.append(aggregated_char_emb)
        text_emb = torch.cat(text_emb_list, 2)
        text_emb = F.dropout(text_emb, self.lexical_dropout)

        text_len_mask = self.sequence_mask(text_len, max_len=max_sentence_length)
        text_len_mask = text_len_mask.view(num_sentences * max_sentence_length)

        text_outputs = self.encode_sentences(text_emb, text_len, text_len_mask)
        text_outputs = F.dropout(text_outputs, self.dropout)

        genre_emb = self.genre_tensor[genre]  # [emb]

        sentence_indices = torch.unsqueeze(torch.arange(num_sentences), 1).repeat(1, max_sentence_length)
        # [num_sentences, max_sentence_length]

        # TODO make sure self.flatten_emb_by_sentence works as expected
        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
        flattened_text_emb = self.flatten_emb_by_sentence(text_emb, text_len_mask)  # [num_words]

        candidate_starts, candidate_ends = coref_ops.coref_kernels_spans(
            sentence_indices=flattened_sentence_indices,
            max_width=self.max_mention_width)

        candidate_mention_emb = self.get_mention_emb(flattened_text_emb, text_outputs, candidate_starts,
                                                     candidate_ends)  # [num_candidates, emb]

        # this is now a nn candidate_mention_scores = self.get_mention_scores(candidate_mention_emb)  # [num_mentions, 1]
        candidate_mention_scores = self.mention(candidate_mention_emb)
        candidate_mention_scores = torch.squeeze(candidate_mention_scores, 1)  # [num_mentions]

        k = int(np.floor(float(text_outputs.shape[0]) * self.config["mention_ratio"]))
        predicted_mention_indices = coref_ops.coref_kernels_extract_mentions(candidate_mention_scores, candidate_starts,
                                                                             candidate_ends, k)  # ([k], [k])
        # predicted_mention_indices.set_shape([None])

        mention_starts = torch.index_select(candidate_starts, 0, predicted_mention_indices.type(torch.LongTensor))  # [num_mentions]
        mention_ends = torch.index_select(candidate_ends, 0, predicted_mention_indices.type(torch.LongTensor))  # [num_mentions]
        mention_emb = torch.index_select(candidate_mention_emb, 0, predicted_mention_indices.type(torch.LongTensor))  # [num_mentions, emb]
        mention_scores = torch.index_select(candidate_mention_scores, 0, predicted_mention_indices.type(torch.LongTensor))  # [num_mentions]

        mention_start_emb = torch.index_select(text_outputs, 0, mention_starts.type(torch.LongTensor))  # [num_mentions, emb]
        mention_end_emb = torch.index_select(text_outputs, 0, mention_ends.type(torch.LongTensor))  # [num_mentions, emb]
        mention_speaker_ids = torch.index_select(speaker_ids, 0, mention_starts.type(torch.LongTensor))  # [num_mentions]

        max_antecedents = self.config["max_antecedents"]
        antecedents, antecedent_labels, antecedents_len = coref_ops.coref_kernels_antecedents(mention_starts,
                                                                                              mention_ends,
                                                                                              gold_starts,
                                                                                              gold_ends,
                                                                                              cluster_ids,
                                                                                              max_antecedents)
        # ([num_mentions, max_ant], [num_mentions, max_ant + 1], [num_mentions]
        antecedent_scores = self.get_antecedent_scores(mention_emb, mention_scores, antecedents, antecedents_len,
                                                       mention_starts, mention_ends, mention_speaker_ids,
                                                       genre_emb)  # [num_mentions, max_ant + 1]
        loss = self.softmax_loss(antecedent_scores, antecedent_labels)  # [num_mentions]
        loss2 = F.multilabel_margin_loss(antecedent_scores, antecedent_labels.type(torch.LongTensor))
        loss = torch.sum(loss)  # []
        return [candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, antecedents,
                antecedent_scores], loss

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + torch.log(antecedent_labels.type(torch.FloatTensor))  # [num_mentions, max_ant + 1]
        marginalized_gold_scores = self.logsumexp(gold_scores, 1, keepdims=True)  # [num_mentions]
        log_norm = self.logsumexp(antecedent_scores, 1, keepdims=True) # [num_mentions]
        return log_norm - marginalized_gold_scores  # [num_mentions]

    def logsumexp(self, x, dim=1, keepdims=True):
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        x = torch.where(
            (xm == float('inf')) | (xm == float('-inf')),
            torch.zeros(xm.shape),
            xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
        return x if keepdims else x.squeeze(dim)

    def reverse_tensor(self, tensor, seq_lengths, seq_dim, batch_dim):
        # this works TODO check if it may also need a split across either seq_dim or batch_dim and of seq_lengths
        idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        inverted_tensor = tensor.index_select(0, idx)
        return inverted_tensor

    def sequence_mask(self, lengths, max_len=None):
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    # text_emb = the 500d embedding of text
    # text_len = length of text
    # text_len_mask = a mask of 0 and 1
    def encode_sentences(self, text_emb, text_len, text_len_mask):
        num_sentences = text_emb.shape[0]
        max_sentence_length = text_emb.shape[1]

        # Transpose before and after because it is expected by the LSTM.
        inputs = torch.transpose(text_emb, 0, 1)  # [max_sentence_length, num_sentences, emb]

        # # with tf.variable_scope("fw_cell"):
        # cell_fw = util.CustomLSTMCell(self.config["lstm_size"], num_sentences, self.dropout)
        # preprocessed_inputs_fw = cell_fw.preprocess_input(inputs)
        # # with tf.variable_scope("bw_cell"):
        # cell_bw = util.CustomLSTMCell(self.config["lstm_size"], num_sentences, self.dropout)
        # preprocessed_inputs_bw = cell_bw.preprocess_input(inputs)
        # # preprocessed_inputs_bw = tf.reverse_sequence(preprocessed_inputs_bw, seq_lengths=text_len, seq_dim=0, batch_dim=1)
        # preprocessed_inputs_bw = self.reverse_tensor(preprocessed_inputs_bw, seq_lengths=text_len, seq_dim=0,
        #                                              batch_dim=1)
        #
        # state_fw = nn.LSTMCell(cell_fw.initial_state.c.repeat(num_sentences, 1),
        #                        cell_fw.initial_state.h.repeat(num_sentences, 1))
        # state_bw = nn.LSTMCell(cell_bw.initial_state.c.repeat([num_sentences, 1]),
        #                        cell_bw.initial_state.h.repeat([num_sentences, 1]))
        # # with tf.variable_scope("lstm"):
        # #     with tf.variable_scope("fw_lstm"):
        # # fw_outputs, fw_states = tf.nn.dynamic_rnn(cell=cell_fw, inputs=preprocessed_inputs_fw, sequence_length=text_len, initial_state=state_fw, time_major=True)
        # fw_outputs, fw_states = cell_fw(preprocessed_inputs_fw, state_fw)
        #
        # # with tf.variable_scope("bw_lstm"):
        # # bw_outputs, bw_states = tf.nn.dynamic_rnn(cell=cell_bw,inputs=preprocessed_inputs_bw,sequence_length=text_len,initial_state=state_bw,time_major=True)
        # bw_outputs, bw_states = cell_bw(preprocessed_inputs_bw, state_bw)
        #
        # # bw_outputs = tf.reverse_sequence(bw_outputs, seq_lengths=text_len, seq_dim=0, batch_dim=1)
        # bw_outputs = self.reverse_tensor(bw_outputs, seq_lengths=text_len, seq_dim=0, batch_dim=1)
        #
        # text_outputs = torch.cat([fw_outputs, bw_outputs], 2)
        self.hidden = self.bilstm_init(self.bilstm.hidden_size, num_sentences)
        text_outputs, self.hidden = self.bilstm(inputs, self.hidden)
        text_outputs = torch.transpose(text_outputs, 0, 1)
        # inputs_list = inputs.chunk(num_sentences, dim=1)
        # text_outputs_list = []
        # for i in range(num_sentences):
        #     text_outputs, self.hidden = self.bilstm(inputs_list[i], self.hidden)
        #     text_outputs_list.append(text_outputs)
        # # [num_sentences, max_sentence_length, emb]
        # text_outputs = torch.transpose(torch.cat(text_outputs_list, dim=1), 0, 1)
        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = emb.shape[0]
        max_sentence_length = emb.shape[1]
        emb_rank = len(emb.shape)
        # TODO check if it works correctly for both rank 2 and 3
        if emb_rank == 2:
            flattened_emb = emb.contiguous().view([num_sentences * max_sentence_length])
            res = torch.masked_select(flattened_emb, text_len_mask.view(-1))
            return res
        elif emb_rank == 3:
            flattened_emb = emb.contiguous().view(num_sentences * max_sentence_length, util.shape(emb, emb_rank - 1))
            res = torch.masked_select(flattened_emb, text_len_mask.view(-1, 1))
            return res.view(-1, util.shape(emb, emb_rank - 1))
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))

    def get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
        mention_emb_list = []

        mention_start_emb = torch.index_select(text_outputs, 0, mention_starts.type(torch.LongTensor))  # [num_mentions, emb]
        mention_emb_list.append(mention_start_emb)

        mention_end_emb = torch.index_select(text_outputs, 0, mention_ends.type(torch.LongTensor))  # [num_mentions, emb]
        mention_emb_list.append(mention_end_emb)

        mention_width = 1 + mention_ends - mention_starts  # [num_mentions]
        if self.config["use_features"]:
            mention_width_index = mention_width - 1  # [num_mentions]
            mention_width_emb = torch.index_select(self.mention_width_tensor, 0, mention_width_index.type(torch.LongTensor))  # [num_mentions, emb]
            mention_width_emb = F.dropout(mention_width_emb, self.dropout)
            mention_emb_list.append(mention_width_emb)

        if self.config["model_heads"]:
            mention_indices = torch.unsqueeze(torch.arange(self.config["max_mention_width"]).type(torch.IntTensor), 0) \
                              + torch.unsqueeze(mention_starts, 1)  # [num_mentions, max_mention_width]
            # replaces the value inside the tensor with the minimum
            min_dim_val = util.shape(text_outputs, 0) - 1
            mention_indices[mention_indices > min_dim_val] = min_dim_val  # [num_mentions, max_mention_width]
            mention_text_emb = torch.index_select(text_emb, 0, mention_indices.type(torch.LongTensor).view(-1)).view(
                mention_indices.shape[0], mention_indices.shape[1], text_emb.shape[1])
            # [num_mentions, max_mention_width, emb]
            head_scores = self.head_scores(text_outputs)  # [num_words, 1]
            mention_head_scores = torch.index_select(head_scores, 0, mention_indices.type(torch.LongTensor).view(-1)).view(
                mention_indices.shape[0], mention_indices.shape[1], 1)
            # [num_mentions, max_mention_width, 1]
            mention_mask = torch.unsqueeze(
                self.sequence_mask(mention_width, self.config["max_mention_width"]).type(torch.FloatTensor),
                2)  # [num_mentions, max_mention_width, 1]
            mention_attention = F.softmax(mention_head_scores + torch.log(mention_mask),
                                          dim=1)  # [num_mentions, max_mention_width, 1]
            mention_head_emb = torch.sum(mention_attention * mention_text_emb, 1)  # [num_mentions, emb]
            mention_emb_list.append(mention_head_emb)

        mention_emb = torch.cat(mention_emb_list, 1)  # [num_mentions, emb]
        return mention_emb

    def get_antecedent_scores(self, mention_emb, mention_scores, antecedents, antecedents_len, mention_starts,
                              mention_ends, mention_speaker_ids, genre_emb):
        num_mentions = util.shape(mention_emb, 0)
        max_antecedents = util.shape(antecedents, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            antecedent_speaker_ids = torch.index_select(mention_speaker_ids, 0, antecedents.view(-1).type(
                torch.LongTensor)).view(num_mentions, max_antecedents)  # [num_mentions, max_ant]
            same_speaker = torch.unsqueeze(mention_speaker_ids, 1) == antecedent_speaker_ids  # [num_mentions, max_ant]

            speaker_pair_emb = torch.index_select(self.same_speaker_emb, 0,
                                                  same_speaker.view(-1).long()).view(num_mentions, max_antecedents, -1)  # [num_mentions, max_ant, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat([num_mentions, max_antecedents, 1])  # [num_mentions, max_ant, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            target_indices = torch.arange(num_mentions)  # [num_mentions]
            mention_distance = torch.unsqueeze(target_indices, 1) - antecedents.type(torch.FloatTensor)  # [num_mentions, max_ant]
            mention_distance_bins = coref_ops.coref_kernels_distance_bins(mention_distance)  # [num_mentions, max_ant]

            mention_distance_emb = torch.index_select(self.mention_distance_emb, 0,
                                                      mention_distance_bins.view(-1).type(
                                                          torch.LongTensor)).view(num_mentions, max_antecedents, -1)  # [num_mentions, max_ant]
            feature_emb_list.append(mention_distance_emb)

        feature_emb = torch.cat(feature_emb_list, 2)  # [num_mentions, max_ant, emb]
        feature_emb = F.dropout(feature_emb, self.dropout)  # [num_mentions, max_ant, emb]

        antecedent_emb = torch.index_select(mention_emb, 0, antecedents.view(-1).type(
            torch.LongTensor)).view(num_mentions, max_antecedents, -1)  # [num_mentions, max_ant, emb]
        target_emb_tiled = torch.unsqueeze(mention_emb, 1).repeat([1, max_antecedents, 1])  # [num_mentions, max_ant, emb]
        similarity_emb = antecedent_emb * target_emb_tiled  # [num_mentions, max_ant, emb]

        pair_emb = torch.cat([target_emb_tiled, antecedent_emb, similarity_emb, feature_emb], 2)
        # [num_mentions, max_ant, emb]

        antecedent_scores = self.antecedent(pair_emb)
        antecedent_scores = torch.squeeze(antecedent_scores, 2)  # [num_mentions, max_ant]

        antecedent_mask = torch.log(
            self.sequence_mask(antecedents_len, max_antecedents).type(torch.FloatTensor))
        antecedent_scores += antecedent_mask  # [num_mentions, max_ant]

        antecedent_scores += torch.unsqueeze(mention_scores, 1) + torch.index_select(
            mention_scores, 0, antecedents.type(torch.LongTensor).view(-1)).view(num_mentions, max_antecedents)
        # [num_mentions, max_ant]
        antecedent_scores = torch.cat([torch.zeros([util.shape(mention_scores, 0), 1]), antecedent_scores], 1)
        # [num_mentions, max_ant + 1]
        return antecedent_scores  # [num_mentions, max_ant + 1]

    #TODO see if all index_select places can call this method instead
    def gather(self, values_tensor, index_tensor, gather_dim, out_dims):
        return torch.index_select(values_tensor,
                                  gather_dim,
                                  index_tensor.type(torch.LongTensor).view(-1)).view(out_dims)


# TODO figure out where to set the dropout and the initialization
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv0 = nn.Conv1d(8, 50, kernel_size=3)
        self.conv1 = nn.Conv1d(8, 50, kernel_size=4)
        self.conv2 = nn.Conv1d(8, 50, kernel_size=5)
        self.pool = nn.MaxPool1d(50)
        self.drop = nn.Dropout(0.5)

    def forward(self, *input):
        x = input[0].transpose(1, 2)
        x1 = F.relu(self.conv0(x))
        x1, _ = torch.max(x1, 2)
        x2 = F.relu(self.conv1(x))
        x2, _ = torch.max(x2, 2)
        x3 = F.relu(self.conv2(x))
        x3, _ = torch.max(x3, 2)
        res = torch.cat([x1, x2, x3], 1)
        return res


class FFNNMention(nn.Module):
    def __init__(self):
        super(FFNNMention, self).__init__()
        self.layer0 = nn.Linear(1320, 150)
        self.layer1 = nn.Linear(150, 150)
        self.layer2 = nn.Linear(150, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, *input):
        x = input[0]
        x = F.relu(self.layer0(x))
        x = self.dropout(x)
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        return x


class FFNNAntecedent(nn.Module):
    def __init__(self):
        super(FFNNAntecedent, self).__init__()
        self.layer0 = nn.Linear(4020, 150)
        self.layer1 = nn.Linear(150, 150)
        self.layer2 = nn.Linear(150, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, *input):
        x = input[0]
        x = F.relu(self.layer0(x))
        x = self.dropout(x)
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        return x


if __name__ == "__main__":
    config = util.get_config("experiments.conf")['best']
    cm = CorefModel(config)
    num_sentences = 3
    max_sentence_length = 5
    text_len_mask = cm.sequence_mask(torch.IntTensor([5, 2, 3]), max_len=5)
    text_len_mask = text_len_mask.view(num_sentences * max_sentence_length)
    sentence_indices = torch.unsqueeze(torch.arange(num_sentences), 1).repeat(1, max_sentence_length).type(torch.IntTensor)
    flattened_sentence_indices = cm.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
