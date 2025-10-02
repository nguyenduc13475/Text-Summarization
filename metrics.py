import re
from collections import defaultdict

import bert_score
import nltk
import ot
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from transformers import logging

logging.set_verbosity_error()

nltk.download("wordnet", quiet=True)


def compute_metric(metrics, candidate_texts, reference_texts, beta=8):
    if not isinstance(metrics, list):
        metrics = [metrics]
    if not isinstance(candidate_texts, list):
        candidate_texts = [candidate_texts]
    if not isinstance(reference_texts, list):
        reference_texts = [reference_texts]
    result = defaultdict(list)
    for metric in metrics:
        # ROUGE-N
        m = re.match(r"^rouge(\d+)$", metric, re.I)
        if m:
            scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
            for candidate_text, reference_text in zip(candidate_texts, reference_texts):
                scores = scorer.score(
                    " ".join(reference_text.split()), " ".join(candidate_text.split())
                )
                result[metric].append(scores[metric].recall)
            continue

        # ROUGE-L
        if metric == "rougeL":
            scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
            for candidate_text, reference_text in zip(candidate_texts, reference_texts):
                scores = scorer.score(
                    " ".join(reference_text.split()), " ".join(candidate_text.split())
                )

                precision = scores[metric].precision
                recall = scores[metric].recall
                if precision + recall == 0:
                    result[metric].append(0.0)
                else:
                    beta2 = beta**2
                    result[metric].append(
                        (1 + beta2) * precision * recall / (beta2 * precision + recall)
                    )
            continue

        # BLEU
        m = re.match(r"^bleu(\d+)$", metric, re.I)
        if m:
            N = int(m.group(1))
            for candidate_text, reference_text in zip(candidate_texts, reference_texts):
                result[metric].append(
                    sentence_bleu(
                        [reference_text.split()],
                        candidate_text.split(),
                        weights=[1 / N] * N,
                        smoothing_function=SmoothingFunction().method1,
                    )
                )
            continue

        # METEOR
        if metric == "meteor":
            for candidate_text, reference_text in zip(candidate_texts, reference_texts):
                result[metric].append(
                    meteor_score([reference_text.split()], candidate_text.split())
                )
            continue

        # BERTScore
        if metric == "bertscore":
            _, _, F1 = bert_score.score(
                candidate_texts, reference_texts, lang="en", verbose=False
            )
            result[metric] = F1.tolist()
            continue

        # MoverScore
        if metric == "moverscore":
            model = SentenceTransformer("all-mpnet-base-v2")

            candidate_embeddings = model.encode(
                ["I have a chicken.", "I have a cock"],
                output_value="token_embeddings",
                convert_to_tensor=True,
            )
            reference_embeddings = model.encode(
                ["I have a chicken.", "I have a bitch."],
                output_value="token_embeddings",
                convert_to_tensor=True,
            )
            candidate_embeddings = [
                F.normalize(candidate_embedding, p=2, dim=1)
                for candidate_embedding in candidate_embeddings
            ]
            reference_embeddings = [
                F.normalize(reference_embedding, p=2, dim=1)
                for reference_embedding in reference_embeddings
            ]

            costs = [
                1 - torch.mm(candidate_embedding, reference_embedding.T)
                for candidate_embedding, reference_embedding in zip(
                    candidate_embeddings, reference_embeddings
                )
            ]
            a_s = [
                torch.ones(candidate_embedding.size(0)) / candidate_embedding.size(0)
                for candidate_embedding in candidate_embeddings
            ]
            b_s = [
                torch.ones(reference_embedding.size(0)) / reference_embedding.size(0)
                for reference_embedding in reference_embeddings
            ]

            emds = [
                ot.emd2(a.numpy(), b.numpy(), cost.detach().cpu().numpy())
                for a, b, cost in zip(a_s, b_s, costs)
            ]
            scores = [1 - emd for emd in emds]
            result[metric] = scores
            continue

        else:
            raise ValueError(f"Metric '{metric}' is not supported.")

    return result
