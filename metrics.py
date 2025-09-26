import re

import bert_score
import nltk
import ot
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer

nltk.download("wordnet")


def compute_metric(metric, candidate_text, reference_text, **kwargs):
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
        beta2 = kwargs["beta"] ** 2
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
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        model = AutoModel.from_pretrained("roberta-large")

        inputs_cand = tokenizer(candidate_text, return_tensors="pt", truncation=True)
        inputs_ref = tokenizer(reference_text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            cand_emb = model(**inputs_cand).last_hidden_state.mean(1).squeeze(0)
            ref_emb = model(**inputs_ref).last_hidden_state.mean(1).squeeze(0)

        cand_emb = cand_emb.unsqueeze(0)
        ref_emb = ref_emb.unsqueeze(0)

        cost = 1 - torch.mm(cand_emb, ref_emb.T) / (
            cand_emb.norm(dim=1, keepdim=True) * ref_emb.norm(dim=1, keepdim=True).T
        )

        w_cand = torch.ones(cand_emb.size(0)) / cand_emb.size(0)
        w_ref = torch.ones(ref_emb.size(0)) / ref_emb.size(0)

        emd = ot.emd2(w_cand.numpy(), w_ref.numpy(), cost.numpy())

        return 1 - emd

    else:
        raise ValueError(f"Metric '{metric}' is not supported.")
