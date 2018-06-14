import torch
from coref_model_pytorch import CorefModel
import util
import coref_ops


def testSpans():
    config = util.get_config("experiments.conf")['best']
    cm = CorefModel(config)
    num_sentences = 3
    max_sentence_length = 5
    text_len_mask = cm.sequence_mask(torch.IntTensor([5, 2, 3]), max_len=5)
    text_len_mask = text_len_mask.view(num_sentences * max_sentence_length)
    sentence_indices = torch.unsqueeze(torch.arange(num_sentences), 1).repeat(1, max_sentence_length).type(torch.IntTensor)
    flattened_sentence_indices = cm.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]

    candidate_starts, candidate_ends = coref_ops.coref_kernels_spans(
        sentence_indices=flattened_sentence_indices,
        max_width=10)
    for (cs, ce) in zip(candidate_starts, candidate_ends):
        print("candidate start: {0} -> end: {1}".format(cs, ce))


if __name__ == "__main__":
    testSpans()