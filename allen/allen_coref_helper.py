import torch
import torch.nn.functional as F

import allen.allen_util as util


def generate_valid_antecedents(num_spans_to_keep: int,
                               max_antecedents: int,
                               device: int):
    """
    This method generates possible antecedents per span which survived the pruning
    stage. This procedure is `generic across the batch`. The reason this is the case is
    that each span in a batch can be coreferent with any previous span, but here we
    are computing the possible `indices` of these spans. So, regardless of the batch,
    the 1st span _cannot_ have any antecedents, because there are none to select from.
    Similarly, each element can only predict previous spans, so this returns a matrix
    of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to
    (i - 1) - j if j <= i, or zero otherwise.

    Parameters
    ----------
    num_spans_to_keep : ``int``, required.
        The number of spans that were kept while pruning.
    max_antecedents : ``int``, required.
        The maximum number of antecedent spans to consider for every span.
    device: ``int``, required.
        The CUDA device to use.

    Returns
    -------
    valid_antecedent_indices : ``torch.IntTensor``
        The indices of every antecedent to consider with respect to the top k spans.
        Has shape ``(num_spans_to_keep, max_antecedents)``.
    valid_antecedent_offsets : ``torch.IntTensor``
        The distance between the span and each of its antecedents in terms of the number
        of considered spans (i.e not the word distance between the spans).
        Has shape ``(1, max_antecedents)``.
    valid_antecedent_log_mask : ``torch.FloatTensor``
        The logged mask representing whether each antecedent span is valid. Required since
        different spans have different numbers of valid antecedents. For example, the first
        span in the document should have no valid antecedents.
        Has shape ``(1, num_spans_to_keep, max_antecedents)``.
    """
    # Shape: (num_spans_to_keep, 1)
    target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

    # Shape: (1, max_antecedents)
    valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

    # This is a broadcasted subtraction.
    # Shape: (num_spans_to_keep, max_antecedents)
    raw_antecedent_indices = target_indices - valid_antecedent_offsets

    # In our matrix of indices, the upper triangular part will be negative
    # because the offsets will be > the target indices. We want to mask these,
    # because these are exactly the indices which we don't want to predict, per span.
    # We're generating a logspace mask here because we will eventually create a
    # distribution over these indices, so we need the 0 elements of the mask to be -inf
    # in order to not mess up the normalisation of the distribution.
    # Shape: (1, num_spans_to_keep, max_antecedents)
    valid_antecedent_log_mask = (raw_antecedent_indices >= 0).float().log()

    # Shape: (num_spans_to_keep, max_antecedents)
    valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
    return valid_antecedent_indices, valid_antecedent_offsets.squeeze(0), valid_antecedent_log_mask


def compute_antecedent_gold_labels(top_span_labels: torch.IntTensor,
                                   antecedent_labels: torch.IntTensor):
    """
    Generates a binary indicator for every pair of spans. This label is one if and
    only if the pair of spans belong to the same cluster. The labels are augmented
    with a dummy antecedent at the zeroth position, which represents the prediction
    that a span does not have any antecedent.

    Parameters
    ----------
    top_span_labels : ``torch.IntTensor``, required.
        The cluster id label for every span. The id is arbitrary,
        as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
    antecedent_labels : ``torch.IntTensor``, required.
        The cluster id label for every antecedent span. The id is arbitrary,
        as we just care about the clustering. Has shape
        (batch_size, num_spans_to_keep, max_antecedents).

    Returns
    -------
    pairwise_labels_with_dummy_label : ``torch.FloatTensor``
        A binary tensor representing whether a given pair of spans belong to
        the same cluster in the gold clustering.
        Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

    """
    # Shape: (batch_size, num_spans_to_keep, max_antecedents)
    top_span_labels = top_span_labels.unsqueeze(0)
    antecedent_labels = antecedent_labels.unsqueeze(0)
    target_labels = top_span_labels.expand_as(antecedent_labels)
    same_cluster_indicator = (target_labels == antecedent_labels).float()
    non_dummy_indicator = (target_labels >= 0).float()
    pairwise_labels = same_cluster_indicator * non_dummy_indicator

    # Shape: (batch_size, num_spans_to_keep, 1)
    dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

    # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
    pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
    return pairwise_labels_with_dummy_label.squeeze(0)
