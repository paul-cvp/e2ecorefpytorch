from collections import Counter
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import torch
torch.set_printoptions(threshold=5000)
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import util
import coref_ops
import conll
import metrics
from allen.allen_endpoint_span_extractor import EndpointSpanExtractor
from allen.allen_self_attentive_span_extractor import SelfAttentiveSpanExtractor
from allen.allen_span_pruner import SpanPruner
from allen.allen_mention_recall import MentionRecall
from allen.allen_conll_coref_scores import ConllCorefScores
import allen.allen_util as aln_util
import allen.allen_coref_helper as aln_coref_helper


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
        self.dropout = nn.Dropout(self.config["dropout_rate"])  # 0.2
        self.lexical_dropout = nn.Dropout(self.config["lexical_dropout_rate"])  # 0.5

        self.char_embeddings = nn.Embedding(115, 8)

        self.char_cnn = CNN()
        self.bilstm = nn.LSTM(input_size=500, hidden_size=200, num_layers=1, dropout=0.2, bidirectional=True)

        self._endpoint_span_extractor = EndpointSpanExtractor(800,
                                                              combination="x,y",
                                                              num_width_embeddings=10,
                                                              span_width_embedding_dim=20,
                                                              bucket_widths=False)
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=400)

        self.genre_emb = nn.Embedding(len(self.genres), self.config["feature_size"])
        # self.mention_width_emb = nn.Embedding(self.config["max_mention_width"], self.config["feature_size"])
        # self.head_scores = nn.Linear(400, 1)
        self.mention = SpanPruner(FFNNMention())
        self.same_speaker_emb = nn.Embedding(2, self.config["feature_size"])
        self.mention_distance_emb = nn.Embedding(10, self.config["feature_size"])
        self.antecedent = FFNNAntecedent()

        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()
        self._regularizer = None

        self.weights_init(self.char_cnn.parameters())
        self.hidden = self.bilstm_init(self.bilstm.hidden_size)
        self.weights_init(self.mention.parameters())
        self.weights_init(self.antecedent.parameters())

        # the coreference score = mention score span 1 + mention score span 2 + pairwise antecedent score of both spans

    @staticmethod
    def bilstm_init(hidden_dim, num_layers=1):
        h_0 = torch.normal(torch.FloatTensor(2, num_layers, hidden_dim))
        c_0 = torch.normal(torch.FloatTensor(2, num_layers, hidden_dim))
        nn.init.orthogonal(h_0)
        nn.init.orthogonal(c_0)
        return Variable(h_0), Variable(c_0)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
            nn.init.xavier_uniform(m.bias.data)

    def forward(self, word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):

        num_sentences = word_emb.shape[0]  # number of sentences to predict from
        max_sentence_length = word_emb.shape[1]  # maybe caused by applying padding to the dataset to have all sentences in the same shape

        text_emb_list = [Variable(word_emb)]  # 3D tensor added in an array

        if self.config["char_embedding_size"] > 0:  # true is 8
            char_emb = self.char_embeddings(Variable(char_index).view(-1,char_index.shape[2])).view([char_index.shape[0],
                                                                                                    char_index.shape[1],
                                                                                                    char_index.shape[2],
                                                                                                    -1])
            # [num_sentences, max_sentence_length, max_word_length, emb]
            # [a vector of embedding 8 for each character for each word for each sentence for all sentences]
            # (according to longest word and longest sentence)

            flattened_char_emb = char_emb.view([num_sentences * max_sentence_length,
                                                util.shape(char_emb, 2),
                                                util.shape(char_emb, 3)])  # [num_sentences * max_sentence_length, max_word_length, emb]

            flattened_aggregated_char_emb = self.char_cnn(flattened_char_emb)  # [num_sentences * max_sentence_length, emb] character level CNN

            aggregated_char_emb = flattened_aggregated_char_emb.view([num_sentences, max_sentence_length,
                                                                      util.shape(flattened_aggregated_char_emb,
                                                                                 1)])  # [num_sentences, max_sentence_length, emb]
            text_emb_list.append(aggregated_char_emb)

        text_emb = torch.cat(text_emb_list, 2)
        text_emb = self.lexical_dropout(text_emb)

        text_len_mask = self.sequence_mask(text_len, max_len=max_sentence_length)
        text_len_mask = text_len_mask.view(num_sentences * max_sentence_length)

        text_outputs = self.encode_sentences(text_emb, text_len_mask)
        text_outputs = self.dropout(text_outputs)

        sentence_indices = torch.unsqueeze(torch.arange(num_sentences), 1).repeat(1, max_sentence_length)  # [num_sentences, max_sentence_length]

        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
        flattened_text_emb = self.flatten_emb_by_sentence(text_emb, text_len_mask)  # [num_words]

        # candidate_starts, candidate_ends = coref_ops.coref_kernels_spans(sentence_indices=flattened_sentence_indices,max_width=self.max_mention_width)
        candidate_starts, candidate_ends, spans = coref_ops.enumerate_spans(flattened_sentence_indices, max_span_width=self.max_mention_width)
        span_labels = self._get_span_labels(gold_starts, gold_ends, cluster_ids, candidate_starts, candidate_ends)

        candidate_starts = F.relu(candidate_starts.float()).long()
        candidate_ends = F.relu(candidate_ends.float()).long()
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        candidate_mention_emb = self._endpoint_span_extractor(text_outputs, candidate_starts, candidate_ends)
        attended_span_embeddings = self._attentive_span_extractor(text_outputs, candidate_starts, candidate_ends)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([candidate_mention_emb, attended_span_embeddings], -1)

        # candidate_mention_emb = self.get_mention_emb(flattened_text_emb, text_outputs, candidate_starts, candidate_ends)  # [num_candidates, emb]

        k = int(np.floor(float(text_outputs.shape[0]) * self.config["mention_ratio"]))

        (top_span_embeddings, top_span_mask, top_span_indices, top_span_mention_scores) = self.mention(span_embeddings, spans, k)
        # candidate_mention_scores = torch.squeeze(candidate_mention_scores, 1)  # [num_mentions]

        #top_span_mask = top_span_mask.unsqueeze(-1)
        num_spans = spans.size(1)
        flat_top_span_indices = aln_util.flatten_and_batch_shift_indices(top_span_indices.unsqueeze(0), num_spans)
        top_spans = util.batched_index_select(spans.unsqueeze(0),
                                              top_span_indices.unsqueeze(0),
                                              flat_top_span_indices).squeeze(0)

        # predicted_mention_indices = coref_ops.coref_kernels_extract_mentions(candidate_mention_scores, candidate_starts,
        #                                                                      candidate_ends, k)  # ([k], [k])
        # predicted_mention_indices.view(-1)

        # mention_starts = torch.index_select(candidate_starts, 0, predicted_mention_indices.type(torch.LongTensor))  # [num_mentions]
        # mention_ends = torch.index_select(candidate_ends, 0, predicted_mention_indices.type(torch.LongTensor))  # [num_mentions]
        # mention_emb = torch.index_select(candidate_mention_emb, 0, predicted_mention_indices.type(torch.LongTensor))  # [num_mentions, emb]
        # mention_scores = torch.index_select(candidate_mention_scores, 0, predicted_mention_indices.type(torch.LongTensor))  # [num_mentions]
        #
        # mention_start_emb = torch.index_select(text_outputs, 0, mention_starts.type(torch.LongTensor))  # [num_mentions, emb]
        # mention_end_emb = torch.index_select(text_outputs, 0, mention_ends.type(torch.LongTensor))  # [num_mentions, emb]
        # mention_speaker_ids = torch.index_select(speaker_ids, 0, mention_starts.type(torch.LongTensor))  # [num_mentions]

        max_antecedents = min(k, self.config["max_antecedents"])
        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = \
            aln_coref_helper.generate_valid_antecedents(k, max_antecedents, aln_util.get_device_of(text_len_mask))

        candidate_antecedent_embeddings = aln_util.flattened_index_select(top_span_embeddings.unsqueeze(0),
                                                                          valid_antecedent_indices).squeeze(0)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        candidate_antecedent_mention_scores = aln_util.flattened_index_select(top_span_mention_scores.unsqueeze(0),
                                                                              valid_antecedent_indices).squeeze(-1).squeeze(0)
        span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings.unsqueeze(0),
                                                                  candidate_antecedent_embeddings.unsqueeze(0),
                                                                  valid_antecedent_offsets.unsqueeze(0)).squeeze(0)

        meta_emb = self._embed_genre_and_speaker(speaker_ids, genre, top_spans.split(1, dim=-1)[0], valid_antecedent_indices)
        if meta_emb is not None:
            span_pair_embeddings = torch.cat([span_pair_embeddings, meta_emb], -1)
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(span_pair_embeddings,
                                                              top_span_mention_scores,
                                                              candidate_antecedent_mention_scores,
                                                              valid_antecedent_log_mask)
        _, predicted_antecedents = coreference_scores.max(1)
        predicted_antecedents -= 1
        # antecedents, antecedent_labels, antecedents_len = coref_ops.coref_kernels_antecedents(mention_starts,
        #                                                                                       mention_ends,
        #                                                                                       gold_starts,
        #                                                                                       gold_ends,
        #                                                                                       cluster_ids,
        #                                                                                       max_antecedents)
        # ([num_mentions, max_ant], [num_mentions, max_ant + 1], [num_mentions]
        # antecedent_scores = self.get_antecedent_scores(mention_emb, mention_scores, antecedents, antecedents_len,
        #                                                mention_starts, mention_ends, mention_speaker_ids,
        #                                                genre_emb)  # [num_mentions, max_ant + 1]
        # loss = self.softmax_loss(antecedent_scores, antecedent_labels)  # [num_mentions]
        # loss = torch.sum(loss)  # []
        # return [candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, antecedents,
        #         antecedent_scores], antecedent_labels, loss

        output_dict = {"top_spans": top_spans,
                       "antecedent_indices": valid_antecedent_indices,
                       "predicted_antecedents": predicted_antecedents}
        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            pruned_gold_labels = util.batched_index_select(span_labels.unsqueeze(0).unsqueeze(-1),
                                                           top_span_indices.unsqueeze(0),
                                                           flat_top_span_indices).squeeze(0)

            antecedent_labels = util.flattened_index_select(pruned_gold_labels.unsqueeze(0),
                                                            valid_antecedent_indices).squeeze(0).squeeze(-1)
            antecedent_labels += valid_antecedent_log_mask.long()
            gold_antecedent_labels = aln_coref_helper.compute_antecedent_gold_labels(pruned_gold_labels, antecedent_labels)
            print(coreference_scores.detach().cpu().data.numpy())
            print(top_span_mask.detach().cpu().data.numpy())
            coreference_log_probs = aln_util.last_dim_log_softmax(coreference_scores.unsqueeze(0), top_span_mask).squeeze(0)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -aln_util.logsumexp(correct_antecedent_log_probs.unsqueeze(0)).sum()

            self._mention_recall(top_spans, gold_starts, gold_ends)
            self._conll_coref_scores(top_spans.data, valid_antecedent_indices.data, predicted_antecedents.data, gold_starts, gold_ends, cluster_ids)

            output_dict["loss"] = negative_marginal_log_likelihood
        return output_dict

    def get_regularization_penalty(self):
        """
        Computes the regularization penalty for the model.
        Returns 0 if the model was not configured to use regularization.
        """
        if self._regularizer is None:
            return 0.0
        else:
            return self._regularizer(self)

    def get_metrics(self, reset: bool = False):
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {"coref_precision": coref_precision,
                "coref_recall": coref_recall,
                "coref_f1": coref_f1,
                "mention_recall": mention_recall}

    @staticmethod
    def _get_span_labels(gold_starts,gold_ends,cluster_ids,candidate_starts,candidate_ends):
        span_labels = []

        cluster_dict = {}
        for cluster_id, gold_start, gold_end in zip(cluster_ids,gold_starts,gold_ends):
            cluster_dict[tuple([gold_start, gold_end])] = cluster_id

        for (start, end) in zip(candidate_starts.detach().data.numpy(), candidate_ends.detach().data.numpy()):
            cand = tuple([start, end])
            if cand in cluster_dict.keys():
                span_labels.append(cluster_dict[cand])
            else:
                span_labels.append(-1)
        return Variable(torch.LongTensor(np.array(span_labels)))

    # @staticmethod
    # def _get_span_labels(gold_starts, gold_ends, cluster_ids, cand_starts, cand_ends):
    #     span_labels = [-1] * gold_starts.shape[0]
    #     cluster_dict = {}
    #     for i, cluster_id in enumerate(cluster_ids):
    #         cluster_dict[tuple([gold_starts[i], gold_ends[i]])] = cluster_id
    #
    #     # for i, (start, end) in enumerate(zip(cand_starts.data.numpy(), cand_ends.data.numpy())):
    #     #     if (start, end) in cluster_dict:
    #     #         span_labels[i] = cluster_dict[(start, end)]
    #     return Variable(torch.from_numpy(np.array(span_labels)))

    def _compute_coreference_scores(self,
                                    pairwise_embeddings: torch.FloatTensor,
                                    top_span_mention_scores: torch.FloatTensor,
                                    antecedent_mention_scores: torch.FloatTensor,
                                    antecedent_log_mask: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        Parameters
        ----------
        pairwise_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of pairs of spans. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, encoding_dim)
        top_span_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every antecedent. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_log_mask: ``torch.FloatTensor``, required.
            The log of the mask for valid antecedents.

        Returns
        -------
        coreference_scores: ``torch.FloatTensor``
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.

        """
        top_span_mention_scores = top_span_mention_scores.unsqueeze(0)
        antecedent_mention_scores = antecedent_mention_scores.unsqueeze(0)
        antecedent_log_mask = antecedent_log_mask.unsqueeze(0)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self.antecedent(pairwise_embeddings).squeeze(-1)
        antecedent_scores = antecedent_scores.unsqueeze(0)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = Variable(antecedent_scores.data.new(*shape).fill_(0), requires_grad=False)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores.squeeze(0)

    def _embed_genre_and_speaker(self,
                                 speaker_ids,
                                 genre,
                                 top_span_starts,
                                 antecedents):
        feature_emb_list = []
        if self.same_speaker_emb is not None:
            speaker_ids = torch.LongTensor(speaker_ids)
            mention_speaker_ids = torch.index_select(speaker_ids, 0, torch.LongTensor(top_span_starts.contiguous().data.cpu().numpy()).view(-1))
            antecedent_speaker_ids = torch.index_select(mention_speaker_ids, 0,
                                                        torch.LongTensor(antecedents.view(-1).data.cpu().numpy())).view(antecedents.shape)
            same_speaker = torch.unsqueeze(mention_speaker_ids, 1) == antecedent_speaker_ids
            speaker_pair_emb = self.same_speaker_emb(Variable(same_speaker.type(torch.LongTensor)))
            feature_emb_list.append(speaker_pair_emb)

        if self.genre_emb is not None:
            genre = torch.LongTensor([genre])
            genre_emb = self.genre_emb(Variable(genre.repeat(antecedents.shape)))
            feature_emb_list.append(genre_emb)

        feature_emb = torch.cat(feature_emb_list, -1)
        feature_emb = self.dropout(feature_emb)

        return feature_emb

    def _compute_span_pair_embeddings(self,
                                      top_span_embeddings: torch.FloatTensor,
                                      antecedent_embeddings: torch.FloatTensor,
                                      antecedent_offsets: torch.FloatTensor):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        Parameters
        ----------
        top_span_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : ``torch.IntTensor``, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (1, max_antecedents).

        Returns
        -------
        span_pair_embeddings : ``torch.FloatTensor``
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self.mention_distance_emb(
            aln_util.bucket_values(antecedent_offsets,
                                   num_total_buckets=10))

        # Shape: (1, 1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)

        expanded_distance_embeddings_shape = (antecedent_embeddings.size(0),
                                              antecedent_embeddings.size(1),
                                              antecedent_embeddings.size(2),
                                              antecedent_distance_embeddings.size(-1))
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(*expanded_distance_embeddings_shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat([target_embeddings,
                                          antecedent_embeddings,
                                          antecedent_embeddings * target_embeddings,
                                          antecedent_distance_embeddings], -1)
        return span_pair_embeddings

    # text_emb = the 500d embedding of text
    # text_len = length of text
    # text_len_mask = a mask of 0 and 1
    def encode_sentences(self, text_emb, text_len_mask):
        num_sentences = text_emb.shape[0]

        # Transpose before and after because it is expected by the LSTM.
        inputs = torch.transpose(text_emb, 0, 1)  # [max_sentence_length, num_sentences, emb]

        self.hidden = self.bilstm_init(self.bilstm.hidden_size, num_sentences)
        text_outputs, self.hidden = self.bilstm(inputs, self.hidden)
        text_outputs = torch.transpose(text_outputs, 0, 1)
        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    @staticmethod
    def flatten_emb_by_sentence(emb, text_len_mask):
        num_sentences = emb.shape[0]
        max_sentence_length = emb.shape[1]
        emb_rank = len(emb.shape)
        if emb_rank == 2:
            flattened_emb = emb.contiguous().view([num_sentences * max_sentence_length])
            res = torch.masked_select(flattened_emb, text_len_mask.data.view(-1).type(torch.ByteTensor))
            return res
        elif emb_rank == 3:
            flattened_emb = emb.contiguous().view(num_sentences * max_sentence_length, util.shape(emb, emb_rank - 1))
            res = torch.masked_select(flattened_emb.data, text_len_mask.data.view(-1, 1))
            return res.view(-1, util.shape(emb, emb_rank - 1))
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))

    @staticmethod
    def sequence_mask(lengths, max_len=None):
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return Variable(torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1)))

    # def softmax_loss(self, antecedent_scores, antecedent_labels):
    #     gold_scores = antecedent_scores.tril() + torch.log(antecedent_labels.type(torch.FloatTensor)).tril()  # [num_mentions, max_ant + 1]
    #     marginalized_gold_scores = self.logsumexp(gold_scores, 1, keepdims=True)  # [num_mentions]
    #     log_norm = self.logsumexp(antecedent_scores, 1, keepdims=True) # [num_mentions]
    #     return log_norm - marginalized_gold_scores  # [num_mentions]
    #
    # @staticmethod
    # def logsumexp(x, dim=1, keepdims=True):
    #     if dim is None:
    #         x, dim = x.view(-1), 0
    #     xm, _ = torch.max(x, dim, keepdim=True)
    #     x = torch.where(
    #         (xm == float('inf')) | (xm == float('-inf')),
    #         torch.zeros(xm.shape),
    #         xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    #     return x if keepdims else x.squeeze(dim)
    #
    # @staticmethod
    # def reverse_tensor(tensor, seq_lengths, seq_dim, batch_dim):
    #     idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    #     idx = torch.LongTensor(idx)
    #     inverted_tensor = tensor.index_select(0, idx)
    #     return inverted_tensor

    # def get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
    #     mention_emb_list = []
    #
    #     mention_start_emb = torch.index_select(text_outputs, 0, mention_starts.type(torch.LongTensor))  # [num_mentions, emb]
    #     mention_emb_list.append(mention_start_emb)
    #
    #     mention_end_emb = torch.index_select(text_outputs, 0, mention_ends.type(torch.LongTensor))  # [num_mentions, emb]
    #     mention_emb_list.append(mention_end_emb)
    #
    #     mention_width = 1 + mention_ends - mention_starts  # [num_mentions]
    #     if self.config["use_features"]:
    #         mention_width_index = mention_width - 1  # [num_mentions]
    #         mention_width_emb = torch.index_select(self.mention_width_emb, 0, mention_width_index.type(torch.LongTensor))  # [num_mentions, emb]
    #         mention_width_emb = F.dropout(mention_width_emb, self.dropout)
    #         mention_emb_list.append(mention_width_emb)
    #
    #     if self.config["model_heads"]:
    #         mention_indices = torch.unsqueeze(torch.arange(self.config["max_mention_width"]).type(torch.IntTensor), 0) \
    #                           + torch.unsqueeze(mention_starts, 1)  # [num_mentions, max_mention_width]
    #         # replaces the value inside the tensor with the minimum
    #         min_dim_val = util.shape(text_outputs, 0) - 1
    #         mention_indices[mention_indices > min_dim_val] = min_dim_val  # [num_mentions, max_mention_width]
    #         mention_text_emb = torch.index_select(text_emb, 0, mention_indices.type(torch.LongTensor).view(-1)).view(
    #             mention_indices.shape[0], mention_indices.shape[1], text_emb.shape[1])  # [num_mentions, max_mention_width, emb]
    #         head_scores = self.head_scores(text_outputs)  # [num_words, 1]
    #         mention_head_scores = torch.index_select(head_scores, 0, mention_indices.type(torch.LongTensor).view(-1)).view(
    #             mention_indices.shape[0], mention_indices.shape[1], 1)  # [num_mentions, max_mention_width, 1]
    #         mention_mask = torch.unsqueeze(
    #             self.sequence_mask(mention_width, self.config["max_mention_width"]).type(torch.FloatTensor),
    #             2)  # [num_mentions, max_mention_width, 1]
    #         mention_attention = F.softmax(mention_head_scores + torch.log(mention_mask),
    #                                       dim=1)  # [num_mentions, max_mention_width, 1]
    #         mention_head_emb = torch.sum(mention_attention * mention_text_emb, 1)  # [num_mentions, emb]
    #         mention_emb_list.append(mention_head_emb)
    #
    #     mention_emb = torch.cat(mention_emb_list, 1)  # [num_mentions, emb]
    #     return mention_emb

    # def get_antecedent_scores(self, mention_emb, mention_scores, antecedents, antecedents_len, mention_starts,
    #                           mention_ends, mention_speaker_ids, genre_emb):
    #     num_mentions = util.shape(mention_emb, 0)
    #     max_antecedents = util.shape(antecedents, 1)
    #
    #     feature_emb_list = []
    #
    #     if self.config["use_metadata"]:
    #         antecedent_speaker_ids = torch.index_select(mention_speaker_ids, 0, antecedents.view(-1).type(
    #             torch.LongTensor)).view(num_mentions, max_antecedents)  # [num_mentions, max_ant]
    #         same_speaker = torch.unsqueeze(mention_speaker_ids, 1) == antecedent_speaker_ids  # [num_mentions, max_ant]
    #
    #         speaker_pair_emb = torch.index_select(self.same_speaker_emb, 0,
    #                                               same_speaker.view(-1).long()).view(num_mentions, max_antecedents, -1)  # [num_mentions, max_ant, emb]
    #         feature_emb_list.append(speaker_pair_emb)
    #
    #         tiled_genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat([num_mentions, max_antecedents, 1])  # [num_mentions, max_ant, emb]
    #         feature_emb_list.append(tiled_genre_emb)
    #
    #     if self.config["use_features"]:
    #         target_indices = torch.arange(num_mentions)  # [num_mentions]
    #         mention_distance = torch.unsqueeze(target_indices, 1) - antecedents.type(torch.FloatTensor)  # [num_mentions, max_ant]
    #         mention_distance_bins = coref_ops.coref_kernels_distance_bins(mention_distance)  # [num_mentions, max_ant]
    #
    #         mention_distance_emb = torch.index_select(self.mention_distance_emb, 0,
    #                                                   mention_distance_bins.view(-1).type(
    #                                                       torch.LongTensor)).view(num_mentions, max_antecedents, -1)  # [num_mentions, max_ant]
    #         feature_emb_list.append(mention_distance_emb)
    #
    #     feature_emb = torch.cat(feature_emb_list, 2)  # [num_mentions, max_ant, emb]
    #     feature_emb = F.dropout(feature_emb, self.dropout)  # [num_mentions, max_ant, emb]
    #
    #     antecedent_emb = torch.index_select(mention_emb, 0, antecedents.view(-1).type(
    #         torch.LongTensor)).view(num_mentions, max_antecedents, -1)  # [num_mentions, max_ant, emb]
    #     target_emb_tiled = torch.unsqueeze(mention_emb, 1).repeat([1, max_antecedents, 1])  # [num_mentions, max_ant, emb]
    #     similarity_emb = antecedent_emb * target_emb_tiled  # [num_mentions, max_ant, emb]
    #
    #     pair_emb = torch.cat([target_emb_tiled, antecedent_emb, similarity_emb, feature_emb], 2)  # [num_mentions, max_ant, emb]
    #
    #     antecedent_scores = self.antecedent(pair_emb)
    #     antecedent_scores = torch.squeeze(antecedent_scores, 2)  # [num_mentions, max_ant]
    #
    #     antecedent_mask = torch.log(
    #         self.sequence_mask(antecedents_len, max_antecedents).type(torch.FloatTensor))
    #     antecedent_scores += antecedent_mask  # [num_mentions, max_ant]
    #
    #     antecedent_scores += torch.unsqueeze(mention_scores, 1) + torch.index_select(
    #         mention_scores, 0, antecedents.type(torch.LongTensor).view(-1)).view(num_mentions, max_antecedents)  # [num_mentions, max_ant]
    #     antecedent_scores = torch.cat([torch.zeros([util.shape(mention_scores, 0), 1]), antecedent_scores], 1)  # [num_mentions, max_ant + 1]
    #     return antecedent_scores  # [num_mentions, max_ant + 1]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv0 = nn.Conv1d(8, 50, kernel_size=3)
        self.conv1 = nn.Conv1d(8, 50, kernel_size=4)
        self.conv2 = nn.Conv1d(8, 50, kernel_size=5)

    def forward(self, input):
        x = input.transpose(1, 2)
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
        self.layer0 = nn.Linear(1220, 150)
        self.acc0 = nn.ReLU()
        self.dropout0 = nn.Dropout(0.2)
        self.layer1 = nn.Linear(150, 150)
        self.acc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(150, 1)
        self.acc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout0(self.acc0(self.layer0(x)))
        x = self.dropout1(self.acc1(self.layer1(x)))
        x = self.dropout2(self.acc2(self.layer2(x)))
        return x


class FFNNAntecedent(nn.Module):
    def __init__(self):
        super(FFNNAntecedent, self).__init__()
        self.layer0 = nn.Linear(3720, 150)
        self.dropout0 = nn.Dropout(0.2)
        self.acc0 = nn.ReLU()
        self.layer1 = nn.Linear(150, 150)
        self.dropout1 = nn.Dropout(0.2)
        self.acc1 = nn.ReLU()
        self.layer2 = nn.Linear(150, 1)
        self.dropout2 = nn.Dropout(0.2)
        self.acc2 = nn.ReLU()

    def forward(self, x):
        x = self.dropout0(self.acc0(self.layer0(x)))
        x = self.dropout1(self.acc1(self.layer1(x)))
        x = self.dropout2(self.acc2(self.layer2(x)))
        return x
