from torch.autograd import Function
import numpy as np
import torch
import functools

# from tensorflow.python import pywrap_tensorflow
# pywrap uses Simplified Wrapper and Interface Generator (SWIG) to encode the c/c++ code
# of the coref_kernels.cc (compiled to coref_kernels.so) into python

# this is a python wrapper of the coref_kernels.cc operations (ops)

# TODO find a way to define coref_kernels.so into torch in python directly
# coref_op_library = tf.load_op_library("./coref_kernels.so")

# To make the op available as a regular function import-able from a Python module, it maybe useful to have the load_op_library call in a Python source file as follows
# spans = coref_op_library.spans
# NotDifferentiable is for ops which do not have gradient. in case they are involved in back prop they automatically propagate zeros
# tf.NotDifferentiable("Spans")

def coref_kernels_spans(sentence_indices, max_width):
    return Spans.apply(sentence_indices, max_width)


class Spans(Function):

    @staticmethod
    def forward(ctx, sentence_indices, max_width):
        starts = []
        ends = []
        spans_list = []
        length = len(sentence_indices)
        for i in range(length):
            j = i
            while j < length and (j - i) < max_width:
                if np.array_equal(sentence_indices[i], sentence_indices[j]):
                    spans_list.append((i, j))
                j += 1

        for span_item in spans_list:
            starts.append(span_item[0])
            ends.append(span_item[1])

        return torch.IntTensor(starts), torch.IntTensor(ends)

    @staticmethod
    def backward(ctx, **kwargs):
        return None


# antecedents = coref_op_library.antecedents
# tf.NotDifferentiable("Antecedents")

def coref_kernels_antecedents(mention_starts, mention_ends, gold_starts, gold_ends, cluster_ids, max_antecedents):
    return Antecedents.apply(mention_starts, mention_ends, gold_starts, gold_ends, cluster_ids, max_antecedents)


class Antecedents(Function):

    @staticmethod
    def forward(ctx, mention_starts, mention_ends, gold_starts, gold_ends, cluster_ids, max_antecedents):

        num_mentions = len(mention_starts)
        num_gold = len(gold_starts)

        max_antecedents = min(num_mentions, max_antecedents)

        antecedents_list = np.zeros((num_mentions, max_antecedents))
        antecedent_labels = np.zeros((num_mentions, max_antecedents+1))
        antecedents_len = np.zeros(num_mentions)
        mention_indices = {}
        for x in range(len(mention_starts)):
            for y in range(len(mention_ends)):
                mention_indices[(x, y)] = 0
        for i in range(num_mentions):
            mention_indices[(int(mention_starts[i]), int(mention_ends[i]))] = i

        mention_cluster_ids = [-1]*num_mentions
        for i in range(num_gold):
            if (int(gold_starts[i]), int(gold_ends[i])) in mention_indices:
                iter = mention_indices[(int(gold_starts[i]), int(gold_ends[i]))]
                mention_cluster_ids[iter] = cluster_ids[i]

        for i in range(num_mentions):
            antecedent_count = 0
            null_label = True
            for j in range(max(0, i - max_antecedents), i):
                if mention_cluster_ids[i] >= 0 and mention_cluster_ids[i] == mention_cluster_ids[j]:
                    antecedent_labels[i][antecedent_count + 1] = True
                    null_label = False
                else:
                    antecedent_labels[i][antecedent_count] = False
                antecedents_list[i][antecedent_count] = j
                antecedent_count += 1
            for j in range(antecedent_count, max_antecedents):
                antecedent_labels[i][j + 1] = False
                antecedents_list[i][j] = 0
            antecedent_labels[i][0] = null_label
            antecedents_len[i] = antecedent_count

        return torch.IntTensor(antecedents_list), torch.ByteTensor(antecedent_labels), torch.IntTensor(antecedents_len)

    @staticmethod
    def backward(ctx, **kwargs):
        return None


# extract_mentions = coref_op_library.extract_mentions
# tf.NotDifferentiable("ExtractMentions")
def coref_kernels_extract_mentions(mention_scores, candidate_starts, candidate_ends, num_output_mentions):
    return ExtractMentions.apply(mention_scores, candidate_starts, candidate_ends, num_output_mentions)


class ExtractMentions(Function):

    @staticmethod
    def forward(ctx, mention_scores, candidate_starts, candidate_ends, num_output_mentions):
        output_mention_indices = []

        def is_crossing(candidate_starts, candidate_ends, i, j):
            s1 = candidate_starts[i]
            s2 = candidate_starts[j]
            e1 = candidate_ends[i]
            e2 = candidate_ends[j]
            return (s1 < s2 and s2 <= e1 and e1 < e2) or (s2 < s1 and s1 <= e2 and e2 < e1)

        num_input_mentions = len(mention_scores)
        sorted_input_mention_indices = [x for x in range(num_input_mentions)]

        def mention_comp(i1, i2):
            return mention_scores[i1] < mention_scores[i2]

        sorted(sorted_input_mention_indices, key=functools.cmp_to_key(mention_comp))
        top_mention_indices = []
        current_mention_index = 0
        while len(top_mention_indices) < num_output_mentions:
            i = sorted_input_mention_indices[current_mention_index]
            any_crossing = False
            for j in top_mention_indices:
                if is_crossing(candidate_starts, candidate_ends, i, j):
                    any_crossing = True
                    break

            if not any_crossing:
                top_mention_indices.append(i)
            current_mention_index += 1

        def candidate_comp(i1, i2):
            if candidate_starts[i1] < candidate_starts[i2]:
                return True
            elif candidate_starts[i1] > candidate_starts[i2]:
                return False
            elif candidate_ends[i1] < candidate_ends[i2]:
                return True
            elif candidate_ends[i1] > candidate_ends[i2]:
                return False
            else:
                return i1 < i2

        sorted(top_mention_indices, key=functools.cmp_to_key(candidate_comp))

        for i in range(num_output_mentions):
            output_mention_indices.append(top_mention_indices[i])

        return torch.IntTensor(output_mention_indices)

    @staticmethod
    def backward(self, *grad_outputs):
        return None


# distance_bins = coref_op_library.distance_bins
# tf.NotDifferentiable("DistanceBins")

def coref_kernels_distance_bins(distances):
    return DistanceBins.apply(distances)

class DistanceBins(Function):

    @staticmethod
    def forward(ctx, distances):

        def get_bin(d):
            if d <= 0:
                return 0
            elif d == 1:
                return 1
            elif d == 2:
                return 2
            elif d == 3:
                return 3
            elif d == 4:
                return 4
            elif d <= 7:
                return 5
            elif d <= 15:
                return 6
            elif d <= 31:
                return 7
            elif d <= 63:
                return 8
            else:
                return 9

        d0 = distances.shape[0]
        d1 = distances.shape[1]
        bins = np.zeros((d0, d1))

        for i in range(d0):
            for j in range(d1):
                bins[i][j] = get_bin(distances[i][j])

        return torch.IntTensor(bins)

    @staticmethod
    def backward(ctx, bins):
        return None
