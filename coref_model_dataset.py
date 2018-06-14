import torch
if torch.cuda.is_available():
    import torch as torch
else:
    import torch as torch
import torch.utils.data as data
import numpy as np
import util
import random
import json

class TrainCorefDataset(data.Dataset):

    def __init__(self, config):
        self.config = config
        self.embedding_info = [(emb["size"], emb["lowercase"]) for emb in config["embeddings"]]
        self.embedding_size = sum(size for size, _ in self.embedding_info)
        self.embedding_dicts = [util.load_embedding_dict(emb["path"], emb["size"], emb["format"]) for emb in
                                config["embeddings"]]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.genres = {g: i for i, g in enumerate(config["genres"])}

        with open(self.config["train_path"]) as f:
            self.train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
        random.shuffle(self.train_examples)
        self.length = len(self.train_examples)


    def __getitem__(self, index):
        example = self.train_examples[index]
        tensorized_example = self.tensorize_example(example, is_training=1)
        #make the numpy arrays into pytorch Tensors (or Variables) to be sent to the neural net
        item = []
        item.append(torch.FloatTensor(tensorized_example[0]))  # Text embeddings. [?,?,350]
        item.append(torch.LongTensor(tensorized_example[1]))  # Character indices.
        item.append(torch.IntTensor(tensorized_example[2]))   # Text lengths.
        item.append(torch.IntTensor(tensorized_example[3]))  # Speaker IDs.
        item.append(tensorized_example[4])  # Genre.
        item.append(tensorized_example[5])   # Is training.
        item.append(torch.IntTensor(tensorized_example[6]))   # Gold starts.
        item.append(torch.IntTensor(tensorized_example[7]))   # Gold ends.
        item.append(torch.IntTensor(tensorized_example[8]))   # Cluster ids.
        return item

    def __len__(self):
        return self.length

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    #creates numpy arrays
    def tensorize_example(self, example, is_training, oov_counts=None):
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util.flatten(example["speakers"])

        assert num_words == len(speakers)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
        word_emb = np.zeros([len(sentences), max_sentence_length, self.embedding_size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        text_len = np.array([len(s) for s in sentences])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                current_dim = 0
                for k, (d, (s, l)) in enumerate(zip(self.embedding_dicts, self.embedding_info)):
                    if l:
                        current_word = word.lower()
                    else:
                        current_word = word
                    if oov_counts is not None and current_word not in d:
                        oov_counts[k] += 1
                    word_emb[i, j, current_dim:current_dim + s] = util.normalize(d[current_word])
                    current_dim += s
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        if is_training == 1 and len(sentences) > self.config["max_training_sentences"]:
            return self.truncate_example(word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts,
                                         gold_ends, cluster_ids)
        else:
            return word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids

    def truncate_example(self, word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                         cluster_ids):
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = word_emb.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences)
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        word_emb = word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        speaker_ids = speaker_ids[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids
