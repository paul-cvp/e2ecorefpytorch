from typing import Tuple
from collections import Counter, OrderedDict
import numpy as np
import torch
from sklearn.utils.linear_assignment_ import linear_assignment


class ConllCorefScores(object):
    def __init__(self) -> None:
        self.scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    def __call__(self, top_spans, antecedent_indices, predicted_antecedents, gold_starts, gold_ends, cluster_ids):
        top_spans, antecedent_indices, predicted_antecedents = self.unwrap_to_tensors(top_spans,
                                                                                      antecedent_indices,
                                                                                      predicted_antecedents)

        gold_clusters, mention_to_gold = self.get_gold_clusters(gold_starts, gold_ends, cluster_ids)
        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_spans,
                                                                               antecedent_indices,
                                                                               predicted_antecedents)
        for scorer in self.scorers:
            scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)

    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
        precision, recall, f1_score = tuple(sum(metric(e) for e in self.scorers) / len(self.scorers)
                                            for metric in metrics)
        if reset:
            self.reset()
        return precision, recall, f1_score

    def reset(self):
        self.scorers = [Scorer(metric) for metric in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @staticmethod
    def get_gold_clusters(gold_starts, gold_ends, cluster_ids):
        gold_clusters = {}
        for i in range(len(cluster_ids)):
            gold_s = gold_starts[i]
            gold_e = gold_ends[i]
            cid = cluster_ids[i]
            if cid in gold_clusters:
                gold_clusters[cid].append(tuple([gold_s, gold_e]))
            else:
                gold_clusters[cid] = [tuple([gold_s, gold_e])]

        gold_clusters = [tuple(tuple(m) for m in gc) for gc in OrderedDict(gold_clusters).values()]
        mention_to_gold = {}
        for gold_cluster in gold_clusters:
            for mention in gold_cluster:
                mention_to_gold[mention] = gold_cluster
        return gold_clusters, mention_to_gold

    @staticmethod
    def get_predicted_clusters(top_spans, antecedent_indices, predicted_antecedents):
        predicted_clusters_to_ids = {}
        clusters = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue

            # Find predicted index in the antecedent spans.
            predicted_index = antecedent_indices[i, predicted_antecedent]
            # Must be a previous span.
            assert i > predicted_index
            antecedent_span = tuple(top_spans[predicted_index])

            # Check if we've seen the span before.
            if antecedent_span in predicted_clusters_to_ids.keys():
                predicted_cluster_id = predicted_clusters_to_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                clusters.append([antecedent_span])
                predicted_clusters_to_ids[antecedent_span] = predicted_cluster_id

            mention = tuple(top_spans[i])
            clusters[predicted_cluster_id].append(mention)
            predicted_clusters_to_ids[mention] = predicted_cluster_id

        # finalise the spans and clusters.
        clusters = [tuple(cluster) for cluster in clusters]
        # Return a mapping of each mention to the cluster containing it.
        predicted_clusters_to_ids = {mention: clusters[cluster_id] for mention, cluster_id
             in predicted_clusters_to_ids.items()}

        return clusters, predicted_clusters_to_ids


class Scorer:
    """
    Mostly borrowed from <https://github.com/clarkkev/deep-coref/blob/master/evaluation.py>
    """
    def __init__(self, metric):
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0
        self.metric = metric

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == self.ceafe:
            p_num, p_den, r_num, r_den = self.metric(predicted, gold)
        else:
            p_num, p_den = self.metric(predicted, mention_to_gold)
            r_num, r_den = self.metric(gold, mention_to_predicted)
        self.precision_numerator += p_num
        self.precision_denominator += p_den
        self.recall_numerator += r_num
        self.recall_denominator += r_den

    def get_f1(self):
        precision = 0 if self.precision_denominator == 0 else \
            self.precision_numerator / float(self.precision_denominator)
        recall = 0 if self.recall_denominator == 0 else \
            self.recall_numerator / float(self.recall_denominator)
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def get_recall(self):
        if self.recall_numerator == 0:
            return 0
        else:
            return self.recall_numerator / float(self.recall_denominator)

    def get_precision(self):
        if self.precision_numerator == 0:
            return 0
        else:
            return self.precision_numerator / float(self.precision_denominator)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    @staticmethod
    def b_cubed(clusters, mention_to_gold):
        """
        Averaged per-mention precision and recall.
        <https://pdfs.semanticscholar.org/cfe3/c24695f1c14b78a5b8e95bcbd1c666140fd1.pdf>
        """
        numerator, denominator = 0, 0
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                if len(cluster2) != 1:
                    correct += count * count
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def muc(clusters, mention_to_gold):
        """
        Counts the mentions in each predicted cluster which need to be re-allocated in
        order for each predicted cluster to be contained by the respective gold cluster.
        <http://aclweb.org/anthology/M/M95/M95-1005.pdf>
        """
        true_p, all_p = 0, 0
        for cluster in clusters:
            all_p += len(cluster) - 1
            true_p += len(cluster)
            linked = set()
            for mention in cluster:
                if mention in mention_to_gold:
                    linked.add(mention_to_gold[mention])
                else:
                    true_p -= 1
            true_p -= len(linked)
        return true_p, all_p

    @staticmethod
    def phi4(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        """
        return 2 * len([mention for mention in gold_clustering if mention in predicted_clustering]) \
               / float(len(gold_clustering) + len(predicted_clustering))

    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.

        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        clusters = [cluster for cluster in clusters if len(cluster) != 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = Scorer.phi4(gold_cluster, cluster)
        matching = linear_assignment(-scores)
        similarity = sum(scores[matching[:, 0], matching[:, 1]])
        return similarity, len(clusters), similarity, len(gold_clusters)
