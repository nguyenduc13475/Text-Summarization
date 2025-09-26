import math
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

        texts = [candidate_text, reference_text]
        idf_dict = {}
        n_docs = len(texts)
        for text in texts:
            tokens = set(tokenizer.tokenize(text))
            for tok in tokens:
                idf_dict[tok] = idf_dict.get(tok, 0) + 1
        for tok, cnt in idf_dict.items():
            idf_dict[tok] = math.log((n_docs + 1) / (cnt + 1))

        tokens_cand = tokenizer.tokenize(candidate_text)
        tokens_ref = tokenizer.tokenize(reference_text)

        inputs_cand = tokenizer(candidate_text, return_tensors="pt", truncation=True)
        inputs_ref = tokenizer(reference_text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            cand_emb = model(**inputs_cand).last_hidden_state.squeeze(0)
            ref_emb = model(**inputs_ref).last_hidden_state.squeeze(0)

        cand_emb = cand_emb / cand_emb.norm(dim=1, keepdim=True)
        ref_emb = ref_emb / ref_emb.norm(dim=1, keepdim=True)

        cost = 1 - torch.mm(cand_emb, ref_emb.T)

        w_cand = torch.tensor([idf_dict.get(tok, 1.0) for tok in tokens_cand])
        w_ref = torch.tensor([idf_dict.get(tok, 1.0) for tok in tokens_ref])

        w_cand = w_cand / w_cand.sum()
        w_ref = w_ref / w_ref.sum()

        emd = ot.emd2(w_cand.numpy(), w_ref.numpy(), cost.numpy())
        return 1 - emd

    else:
        raise ValueError(f"Metric '{metric}' is not supported.")
