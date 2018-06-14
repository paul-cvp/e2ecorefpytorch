import operator
import random
import math
import json
import threading
import numpy as np
# import tensorflow as tf
import torch as tf
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch import autograd

import util
import coref_ops
import conll
import metrics


class CorefModel(nn.Module):
    def forward(self, *input):
        return self.get_predictions_and_loss(input)

    def __init__(self, config):
        self.config = config
        self.embedding_info = [(emb["size"], emb["lowercase"]) for emb in config["embeddings"]] #[(300,false)(50,false)]
        self.embedding_size = sum(size for size, _ in self.embedding_info) #350 = 300+50
        self.char_embedding_size = config["char_embedding_size"] #8
        self.char_dict = util.load_char_dict(config["char_vocab_path"]) #all characters + <unk> size 115
        self.embedding_dicts = [util.load_embedding_dict(emb["path"], emb["size"], emb["format"]) for emb in
                                config["embeddings"]] #dictionary [(43994?,300)(268822,50)]
        self.max_mention_width = config["max_mention_width"] #10
        self.genres = {g: i for i, g in enumerate(config["genres"])} #types of corpus documents
        #(news = nw, conversational telephone speech=tc, weblogs=wb, usenet newsgroups, broadcast=bc, talk shows)
        #[bc, bn, mz, nw, pt, tc, wb]
        self.eval_data = None  # Load eval data lazily.

        input_props = []
        input_props.append((tf.FloatTensor, [None, None, self.embedding_size]))  # Text embeddings. [?,?,350]
        input_props.append((tf.IntTensor, [None, None, None]))  # Character indices.
        input_props.append((tf.IntTensor, [None]))  # Text lengths.
        input_props.append((tf.IntTensor, [None]))  # Speaker IDs.
        input_props.append((tf.IntTensor, []))      # Genre.
        input_props.append((tf.ByteTensor, []))     # Is training.
        input_props.append((tf.IntTensor, [None]))  # Gold starts.
        input_props.append((tf.IntTensor, [None]))  # Gold ends.
        input_props.append((tf.IntTensor, [None]))  # Cluster ids.
        self.queue_input_tensors = [tf.zeros(shape).type(dtype) for dtype, shape in input_props]
        # dtypes, shapes = zip(*input_props)
        # queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        # self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        # self.input_tensors = queue.dequeue()
        self.input_tensors = self.queue_input_tensors #9 items from input_props that are split when calling get_prediction_and_loss
        # this is the training step more or less
        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)

        self.global_step = tf.zeros()#.Variable(0, name="global_step", trainable=False)
        # self.reset_global_step = tf.assign(self.global_step, 0)

        #here you update something based on yout prediction and loss
        trainable_params = autograd.Variable(0) #this is equivalent to model.parameters() tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params) #this is autograd backward pass
        # Constructs symbolic derivatives of sum of self.loss w.r.t. x in trainable_params
        gradients, _ = nn.utils.clip_grad_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": optim.Adam(trainable_params, lr=self.config["learning_rate"], weight_decay=self.config["decay_rate"]),
            "sgd": optim.SGD(trainable_params, lr=self.config["learning_rate"], weight_decay=self.config["decay_rate"])
        }
        optimizer = optimizers[self.config["optimizer"]]

        learning_rate = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config["decay_frequency"])
        learning_rate.step()
        #self.config["learning_rate"], self.global_step, self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
        # self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)
        # still part of autograd backward
        # minimize = gradients -> apply_gradients
        # here we have gradients -> clip norm -> apply_gradients

    #the input_tensor split into parts
    def get_predictions_and_loss(self, word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts,
                                 gold_ends, cluster_ids):
        training_num = 0.0
        if is_training:
            training_num = 1.0

        #set the dropout rate
        self.dropout = 1 - (training_num * self.config["dropout_rate"]) # 0.2
        self.lexical_dropout = 1 - (training_num * self.config["lexical_dropout_rate"]) # 0.5

        #get the size of tensors num of sentences and max sentence length
        num_sentences = word_emb.shape[0] # number of sentences to predict from
        max_sentence_length = word_emb.shape[1]
        #there is a padding to the dataset to have all sentences in the same shape

        text_emb_list = [word_emb] #3D tensor added in an array the 350D word embedding from glove and turian

        if self.config["char_embedding_size"] > 0: #true is 8
            temp_tensor = tf.zeros([len(self.char_dict), self.config["char_embedding_size"]]) # [115,8]
            nn.init.xavier_uniform(temp_tensor)
            char_emb = tf.gather(temp_tensor, char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
            #[a vector of embedding 8 for each character for each word for each sentence for all sentences]
            # (according to longest word and longest sentence)

            flattened_char_emb = char_emb.view([num_sentences * max_sentence_length,
                                                util.shape(char_emb, 2),
                                                util.shape(char_emb, 3)])
            # [num_sentences * max_sentence_length, max_word_length, emb]

            flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config[
                "filter_size"])  # [num_sentences * max_sentence_length, emb] character level CNN
            aggregated_char_emb = flattened_aggregated_char_emb.view([num_sentences, max_sentence_length,
                                                                      util.shape(flattened_aggregated_char_emb,
                                                                                 1)])  # [num_sentences, max_sentence_length, emb]
            text_emb_list.append(aggregated_char_emb)
        #text_emb_list has 3D tensors 350D word embeddings + 150D character embeddings 50 for each 3, 4, 5 filter size and each character is an 8dim vector
        text_emb = tf.cat(text_emb_list, 2) #concatenated on the second dimension
        text_emb = F.dropout(text_emb, self.lexical_dropout)

        text_len_mask = self.sequence_mask(text_len, max_len=max_sentence_length)
        #tf.sequence_mask(text_len, maxlen=max_sentence_length)
        text_len_mask = text_len_mask.view([num_sentences * max_sentence_length])

        text_outputs = self.encode_sentences(text_emb, text_len, text_len_mask)
        text_outputs = F.dropout(text_outputs, self.dropout)

        genre_tensor = tf.zeros([len(self.genres), self.config["feature_size"]])
        nn.init.xavier_uniform(genre_tensor)
        genre_emb = tf.gather(genre_tensor, genre)  # [emb]

        sentence_indices = tf.unsqueeze(tf.range(num_sentences), 1).repeat(
            [1, max_sentence_length])  # [num_sentences, max_sentence_length]
        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
        flattened_text_emb = self.flatten_emb_by_sentence(text_emb, text_len_mask)  # [num_words]

        candidate_starts, candidate_ends = coref_ops.coref_kernels_spans(
            sentence_indices=flattened_sentence_indices,
            max_width=self.max_mention_width)
        candidate_starts.set_shape([None])
        candidate_ends.set_shape([None])

        candidate_mention_emb = self.get_mention_emb(flattened_text_emb, text_outputs, candidate_starts,
                                                     candidate_ends)  # [num_candidates, emb]
        candidate_mention_scores = self.get_mention_scores(candidate_mention_emb)  # [num_mentions, 1]
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [num_mentions]

        k = tf.floor((text_outputs.shape[0].float()) * self.config["mention_ratio"]).int()
        predicted_mention_indices = coref_ops.coref_kernels_extract_mentions(candidate_mention_scores, candidate_starts,
                                                                             candidate_ends, k)  # ([k], [k])
        predicted_mention_indices.set_shape([None])

        mention_starts = tf.gather(candidate_starts, predicted_mention_indices)  # [num_mentions]
        mention_ends = tf.gather(candidate_ends, predicted_mention_indices)  # [num_mentions]
        mention_emb = tf.gather(candidate_mention_emb, predicted_mention_indices)  # [num_mentions, emb]
        mention_scores = tf.gather(candidate_mention_scores, predicted_mention_indices)  # [num_mentions]

        mention_start_emb = tf.gather(text_outputs, mention_starts)  # [num_mentions, emb]
        mention_end_emb = tf.gather(text_outputs, mention_ends)  # [num_mentions, emb]
        mention_speaker_ids = tf.gather(speaker_ids, mention_starts)  # [num_mentions]

        max_antecedents = self.config["max_antecedents"]
        antecedents, antecedent_labels, antecedents_len = coref_ops.coref_kernels_antecedents(mention_starts,
                                                                                              mention_ends,
                                                                                              gold_starts, gold_ends,
                                                                                              cluster_ids,
                                                                                              max_antecedents)  # ([num_mentions, max_ant], [num_mentions, max_ant + 1], [num_mentions]
        antecedents.set_shape([None, None])
        antecedent_labels.set_shape([None, None])
        antecedents_len.set_shape([None])

        antecedent_scores = self.get_antecedent_scores(mention_emb, mention_scores, antecedents, antecedents_len,
                                                       mention_starts, mention_ends, mention_speaker_ids,
                                                       genre_emb)  # [num_mentions, max_ant + 1]

        loss = self.softmax_loss(antecedent_scores, antecedent_labels)  # [num_mentions]
        loss = tf.sum(loss)  # []

        return [candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, antecedents,
                antecedent_scores], loss

    #this is the loading of data + tensorize example and mentions
    def start_enqueue_thread(self):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                for example in train_examples:
                    tensorized_example = self.tensorize_example(example, is_training=True)
                    feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                    # session.run(self.enqueue_op, feed_dict=feed_dict) TODO use autograd somehow

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (tf.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
        mention_emb_list = []

        mention_start_emb = tf.gather(text_outputs, mention_starts)  # [num_mentions, emb]
        mention_emb_list.append(mention_start_emb)

        mention_end_emb = tf.gather(text_outputs, mention_ends)  # [num_mentions, emb]
        mention_emb_list.append(mention_end_emb)

        mention_width = 1 + mention_ends - mention_starts  # [num_mentions]
        if self.config["use_features"]:
            mention_width_index = mention_width - 1  # [num_mentions]
            temp_tensor = tf.zeros([self.config["max_mention_width"], self.config["feature_size"]])
            nn.init.xavier_uniform(temp_tensor)
            mention_width_emb = tf.gather(temp_tensor, mention_width_index)  # [num_mentions, emb]
            mention_width_emb = F.dropout(mention_width_emb, self.dropout)
            mention_emb_list.append(mention_width_emb)

        if self.config["model_heads"]:
            mention_indices = tf.unsqueeze(tf.range(self.config["max_mention_width"]), 0) + tf.unsqueeze(
                mention_starts, 1)  # [num_mentions, max_mention_width]
            mention_indices = tf.min((util.shape(text_outputs, 0) - 1),
                                     mention_indices)  # [num_mentions, max_mention_width]
            mention_text_emb = tf.gather(text_emb, mention_indices)  # [num_mentions, max_mention_width, emb]
            self.head_scores = util.projection(text_outputs, 1)  # [num_words, 1]
            mention_head_scores = tf.gather(self.head_scores, mention_indices)  # [num_mentions, max_mention_width, 1]
            mention_mask = tf.unsqueeze(
                tf.sequence_mask(mention_width, self.config["max_mention_width"], dtype=tf.float32),
                2)  # [num_mentions, max_mention_width, 1]
            mention_attention = F.softmax(mention_head_scores + tf.log(mention_mask),
                                          dim=1)  # [num_mentions, max_mention_width, 1]
            mention_head_emb = tf.sum(mention_attention * mention_text_emb, 1)  # [num_mentions, emb]
            mention_emb_list.append(mention_head_emb)

        mention_emb = tf.cat(mention_emb_list, 1)  # [num_mentions, emb]
        return mention_emb

    def get_mention_scores(self, mention_emb):
        # with tf.variable_scope("mention_scores"):
        return util.ffnn(mention_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                         self.dropout)  # [num_mentions, 1]

    def to_scalar(self, var):
        # returns a python float
        return var.view(-1).data.tolist()[0]

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = tf.max(vec, 1)
        return self.to_scalar(idx)

    # Compute log sum exp in a numerically stable way for the forward algorithm
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
               tf.log(tf.sum(tf.exp(vec - max_score_broadcast)))

    # version 2 of logsumexp
    def logsumexp(self, tensor, dim=1, keepdim=False):
        max_score, _ = tensor.max(dim, keepdim=keepdim)
        if keepdim:
            stable_vec = tensor - max_score
        else:
            stable_vec = tensor - max_score.unsqueeze(dim)
        return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(antecedent_labels.type(tf.FloatTensor))  # [num_mentions, max_ant + 1]
        marginalized_gold_scores = self.log_sum_exp(
            gold_scores)  # tf.reduce_logsumexp(gold_scores, [1])  # [num_mentions]
        log_norm = self.log_sum_exp(antecedent_scores)  # tf.reduce_logsumexp(antecedent_scores, [1])  # [num_mentions]
        return log_norm - marginalized_gold_scores  # [num_mentions]

    def get_antecedent_scores(self, mention_emb, mention_scores, antecedents, antecedents_len, mention_starts,
                              mention_ends, mention_speaker_ids, genre_emb):
        num_mentions = util.shape(mention_emb, 0)
        max_antecedents = util.shape(antecedents, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            antecedent_speaker_ids = tf.gather(mention_speaker_ids, antecedents)  # [num_mentions, max_ant]
            same_speaker = tf.equal(tf.unsqueeze(mention_speaker_ids, 1),
                                    antecedent_speaker_ids)  # [num_mentions, max_ant]
            same_speaker_emb = tf.zeros([2, self.config["feature_size"]])
            nn.init.xavier_uniform(same_speaker_emb)
            speaker_pair_emb = tf.gather(same_speaker_emb, same_speaker.int())  # [num_mentions, max_ant, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.unsqueeze(tf.unsqueeze(genre_emb, 0), 0).repeat(
                [num_mentions, max_antecedents, 1])  # [num_mentions, max_ant, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            target_indices = tf.range(num_mentions)  # [num_mentions]
            mention_distance = tf.unsqueeze(target_indices, 1) - antecedents  # [num_mentions, max_ant]
            mention_distance_bins = coref_ops.cofer_kernels_distance_bins(mention_distance)  # [num_mentions, max_ant]
            mention_distance_bins.set_shape([None, None])
            mention_distance_emb = tf.zeros([10, self.config["feature_size"]])
            nn.init.xavier_uniform(mention_distance_emb)
            mention_distance_emb = tf.gather(mention_distance_emb, mention_distance_bins)  # [num_mentions, max_ant]
            feature_emb_list.append(mention_distance_emb)

        feature_emb = tf.cat(feature_emb_list, 2)  # [num_mentions, max_ant, emb]
        feature_emb = F.dropout(feature_emb, self.dropout)  # [num_mentions, max_ant, emb]

        antecedent_emb = tf.gather(mention_emb, antecedents)  # [num_mentions, max_ant, emb]
        target_emb_tiled = tf.unsqueeze(mention_emb, 1).repeat([1, max_antecedents, 1])  # [num_mentions, max_ant, emb]
        similarity_emb = antecedent_emb * target_emb_tiled  # [num_mentions, max_ant, emb]

        pair_emb = tf.cat([target_emb_tiled, antecedent_emb, similarity_emb, feature_emb],
                          2)  # [num_mentions, max_ant, emb]

        # with tf.variable_scope("iteration"):
        #     with tf.variable_scope("antecedent_scoring"):
        antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                      self.dropout)  # [num_mentions, max_ant, 1]
        antecedent_scores = tf.squeeze(antecedent_scores, 2)  # [num_mentions, max_ant]

        antecedent_mask = tf.log(
            tf.sequence_mask(antecedents_len, max_antecedents, dtype=tf.float32))  # [num_mentions, max_ant]
        antecedent_scores += antecedent_mask  # [num_mentions, max_ant]

        antecedent_scores += tf.unsqueeze(mention_scores, 1) + tf.gather(mention_scores,
                                                                         antecedents)  # [num_mentions, max_ant]
        antecedent_scores = tf.cat([tf.zeros([util.shape(mention_scores, 0), 1]), antecedent_scores],
                                   1)  # [num_mentions, max_ant + 1]
        return antecedent_scores  # [num_mentions, max_ant + 1]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = emb.shape[0]
        max_sentence_length = emb.shape[1]

        emb_rank = len(emb.shape)
        if emb_rank == 2:
            flattened_emb = emb.view([num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = emb.view([num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, text_len_mask)

    def reverse_tensor(self, tensor, seq_lengths, seq_dim, batch_dim):
        # TODO check if it may also need a split across either seq_dim or batch_dim and of seq_lengths
        idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        idx = tf.LongTensor(idx)
        inverted_tensor = tensor.index_select(0, idx)
        return inverted_tensor

    # text_emb = the 500d embedding of text
    # text_len = length of text
    # text_len_mask = a mask of 0 and 1
    def encode_sentences(self, text_emb, text_len, text_len_mask):
        num_sentences = text_emb.shape[0]
        max_sentence_length = text_emb.shape[1]

        # Transpose before and after for efficiency.
        inputs = tf.transpose(text_emb, [1, 0, 2])  # [max_sentence_length, num_sentences, emb]

        # with tf.variable_scope("fw_cell"):
        cell_fw = util.CustomLSTMCell(self.config["lstm_size"], num_sentences, self.dropout)
        preprocessed_inputs_fw = cell_fw.preprocess_input(inputs)
        # with tf.variable_scope("bw_cell"):
        cell_bw = util.CustomLSTMCell(self.config["lstm_size"], num_sentences, self.dropout)
        preprocessed_inputs_bw = cell_bw.preprocess_input(inputs)
        # preprocessed_inputs_bw = tf.reverse_sequence(preprocessed_inputs_bw, seq_lengths=text_len, seq_dim=0, batch_dim=1)
        preprocessed_inputs_bw = self.reverse_tensor(preprocessed_inputs_bw, seq_lengths=text_len, seq_dim=0,
                                                     batch_dim=1)

        state_fw = nn.LSTMCell(cell_fw.initial_state.c.repeat(num_sentences, 1),
                               cell_fw.initial_state.h.repeat(num_sentences, 1))
        state_bw = nn.LSTMCell(cell_bw.initial_state.c.repeat([num_sentences, 1]),
                               cell_bw.initial_state.h.repeat([num_sentences, 1]))
        # with tf.variable_scope("lstm"):
        #     with tf.variable_scope("fw_lstm"):
        # fw_outputs, fw_states = tf.nn.dynamic_rnn(cell=cell_fw, inputs=preprocessed_inputs_fw, sequence_length=text_len, initial_state=state_fw, time_major=True)
        fw_outputs, fw_states = cell_fw(preprocessed_inputs_fw, state_fw)

        # with tf.variable_scope("bw_lstm"):
        # bw_outputs, bw_states = tf.nn.dynamic_rnn(cell=cell_bw,inputs=preprocessed_inputs_bw,sequence_length=text_len,initial_state=state_bw,time_major=True)
        bw_outputs, bw_states = cell_bw(preprocessed_inputs_bw, state_bw)

        # bw_outputs = tf.reverse_sequence(bw_outputs, seq_lengths=text_len, seq_dim=0, batch_dim=1)
        bw_outputs = self.reverse_tensor(bw_outputs, seq_lengths=text_len, seq_dim=0, batch_dim=1)

        text_outputs = tf.cat([fw_outputs, bw_outputs], 2)
        text_outputs = tf.transpose(text_outputs, [1, 0, 2])  # [num_sentences, max_sentence_length, emb]
        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def evaluate_mentions(self, candidate_starts, candidate_ends, mention_starts, mention_ends, mention_scores,
                          gold_starts, gold_ends, example, evaluators):
        text_length = sum(len(s) for s in example["sentences"])
        gold_spans = set(zip(gold_starts, gold_ends))

        if len(candidate_starts) > 0:
            sorted_starts, sorted_ends, _ = zip(
                *sorted(zip(candidate_starts, candidate_ends, mention_scores), key=operator.itemgetter(2),
                        reverse=True))
        else:
            sorted_starts = []
            sorted_ends = []

        for k, evaluator in evaluators.items():
            if k == -3:
                predicted_spans = set(zip(candidate_starts, candidate_ends)) & gold_spans
            else:
                if k == -2:
                    predicted_starts = mention_starts
                    predicted_ends = mention_ends
                elif k == 0:
                    is_predicted = mention_scores > 0
                    predicted_starts = candidate_starts[is_predicted]
                    predicted_ends = candidate_ends[is_predicted]
                else:
                    if k == -1:
                        num_predictions = len(gold_spans)
                    else:
                        num_predictions = (k * text_length) / 100
                    predicted_starts = sorted_starts[:num_predictions]
                    predicted_ends = sorted_ends[:num_predictions]
                predicted_spans = set(zip(predicted_starts, predicted_ends))
            evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, mention_starts, mention_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(mention_starts[predicted_index]), int(mention_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(mention_starts[i]), int(mention_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, mention_starts, mention_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(mention_starts, mention_ends,
                                                                               predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):
        if self.eval_data is None:
            oov_counts = [0 for _ in self.embedding_dicts]
            with open(self.config["eval_path"]) as f:
                self.eval_data = map(lambda example: (
                    self.tensorize_example(example, is_training=False, oov_counts=oov_counts), example),
                                     (json.loads(jsonline) for jsonline in f.readlines()))
            num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
            for emb, c in zip(self.config["embeddings"], oov_counts):
                print("OOV rate for {}: {:.2f}%".format(emb["path"], (100.0 * c) / num_words))
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, session, official_stdout=False):
        self.load_eval_data()

        def _k_to_tag(k):
            if k == -3:
                return "oracle"
            elif k == -2:
                return "actual"
            elif k == -1:
                return "exact"
            elif k == 0:
                return "threshold"
            else:
                return "{}%".format(k)

        mention_evaluators = {k: util.RetrievalEvaluator() for k in [-3, -2, -1, 0, 10, 15, 20, 25, 30, 40, 50]}

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            candidate_starts, candidate_ends, mention_scores, mention_starts, mention_ends, antecedents, antecedent_scores = session.run(
                self.predictions, feed_dict=feed_dict)

            self.evaluate_mentions(candidate_starts, candidate_ends, mention_starts, mention_ends, mention_scores,
                                   gold_starts, gold_ends, example, mention_evaluators)
            predicted_antecedents = self.get_predicted_antecedents(antecedents, antecedent_scores)

            coref_predictions[example["doc_key"]] = self.evaluate_coref(mention_starts, mention_ends,
                                                                        predicted_antecedents, example["clusters"],
                                                                        coref_evaluator)

            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

        summary_dict = {}
        for k, evaluator in sorted(mention_evaluators.items(), key=operator.itemgetter(0)):
            tags = ["{} @ {}".format(t, _k_to_tag(k)) for t in ("R", "P", "F")]
            results_to_print = []
            for t, v in zip(tags, evaluator.metrics()):
                results_to_print.append("{:<10}: {:.2f}".format(t, v))
                summary_dict[t] = v
            print(", ".join(results_to_print))

        conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        summary_dict["Average F1 (conll)"] = average_f1
        print("Average F1 (conll): {:.2f}%".format(average_f1))

        p, r, f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}%".format(f * 100))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        return util.make_summary(summary_dict), average_f1
