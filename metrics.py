import re

import bert_score
import nltk
import ot
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F

nltk.download("wordnet")


def compute_metric(metric, candidate_text, reference_text, beta=8):
    # ROUGE-N
    m = re.match(r"^rouge(\d+)$", metric, re.I)
    if m:
        N = int(m.group(1))
        if not (1 <= N <= 9):
            raise ValueError(f"rouge_score only supports ROUGE-N with N from 1 to 9")

        scorer = rouge_scorer.RougeScorer([metric.lower()], use_stemmer=True)
        scores = scorer.score(
            " ".join(reference_text.split()), " ".join(candidate_text.split())
        )
        return scores[metric.lower()].recall

    # ROUGE-L
    if metric == "rougeL":
        scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
        scores = scorer.score(
            " ".join(reference_text.split()), " ".join(candidate_text.split())
        )

        precision = scores[metric].precision
        recall = scores[metric].recall
        if precision + recall == 0:
            return 0.0
        beta2 = beta**2
        return (1 + beta2) * precision * recall / (beta2 * precision + recall)

    # BLEU
    m = re.match(r"^bleu(\d+)$", metric, re.I)
    if m:
        N = int(m.group(1))
        return sentence_bleu(
            [reference_text.split()],
            candidate_text.split(),
            weights=[1 / N] * N,
            # smoothing to avoid log 0 (add every where)
            smoothing_function=SmoothingFunction().method1,
        )

    # METEOR
    if metric == "meteor":
        return meteor_score([reference_text.split()], candidate_text.split())

    # BERTScore
    if metric == "bertscore":
        _, _, F1 = bert_score.score(
            [candidate_text], [reference_text], lang="en", verbose=False
        )
        return float(F1.mean())

    # MoverScore
    if metric == "moverscore":
        model = SentenceTransformer("all-mpnet-base-v2")

        candidate_embedding = model.encode(
            candidate_text,
            output_value="token_embeddings",
            convert_to_tensor=True,
        )
        reference_embedding = model.encode(
            reference_text,
            output_value="token_embeddings",
            convert_to_tensor=True,
        )

        candidate_embedding = F.normalize(candidate_embedding, p=2, dim=1)
        reference_embedding = F.normalize(reference_embedding, p=2, dim=1)

        cost = 1 - torch.mm(candidate_embedding, reference_embedding.T)

        a = torch.ones(candidate_embedding.size(0)) / candidate_embedding.size(0)
        b = torch.ones(reference_embedding.size(0)) / reference_embedding.size(0)

        emd = ot.emd2(a.numpy(), b.numpy(), cost.detach().cpu().numpy())
        score = 1 - emd
        return score

    else:
        raise ValueError(f"Metric '{metric}' is not supported.")
