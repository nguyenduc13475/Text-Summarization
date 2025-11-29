import torch


class BeamSearch:
    def __init__(self, batch_size, beam_width, start_token, end_token, device):
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.start_token = start_token
        self.end_token = end_token
        self.device = device

        self.sequences = torch.full(
            (batch_size * beam_width, 1),
            start_token,
            dtype=torch.long,
            device=device,
        )

        self.scores = None
        self.finishes = None

    def init_from_first_topk(self, batch_final_distributions):
        batch_log_probs = torch.log(batch_final_distributions + 1e-9)
        topk_vals, topk_idxs = torch.topk(batch_log_probs, k=self.beam_width, dim=1)
        chosen_tokens = topk_idxs.view(-1)
        self.sequences = torch.cat([self.sequences, chosen_tokens.unsqueeze(1)], dim=1)
        self.scores = topk_vals.view(-1).clone()
        self.finishes = (chosen_tokens == self.end_token).view(-1)
        return chosen_tokens

    def advance(
        self,
        batch_final_distributions,
        trigram_penalty=-30,
        bigram_penalty=-10,
        unigram_penalty=-2,
        penalty_range=15,
    ):
        batch_log_probs = torch.log(batch_final_distributions + 1e-9)
        if self.finishes.any():
            finished_rows = self.finishes.nonzero(as_tuple=False).squeeze(1)
            if finished_rows.numel() > 0:
                batch_log_probs[finished_rows] = -1e9
                batch_log_probs[finished_rows, self.end_token] = 0.0

        seqs = self.sequences.tolist()
        for i, seq in enumerate(seqs):
            if len(seq) >= 3:
                trigrams = set(tuple(seq[j : j + 3]) for j in range(len(seq) - 2))
                last_two = tuple(seq[-2:])
                for token in range(batch_log_probs.shape[1]):
                    if last_two + (token,) in trigrams:
                        batch_log_probs[i, token] += trigram_penalty
            if len(seq) >= 2:
                recent_bigrams = [
                    tuple(seq[j : j + 2])
                    for j in range(max(0, len(seq) - penalty_range), len(seq) - 1)
                ]
                last_token = seq[-1]
                for token in range(batch_log_probs.shape[1]):
                    if (last_token, token) in recent_bigrams:
                        batch_log_probs[i, token] += bigram_penalty
            if len(seq) >= 1:
                for token in set(seq[-penalty_range:]):
                    batch_log_probs[i, token] += unigram_penalty

        topk_vals, topk_idxs = torch.topk(batch_log_probs, k=self.beam_width, dim=1)

        cand_scores = self.scores.unsqueeze(1) + topk_vals
        cand_scores_per_input = cand_scores.view(
            self.batch_size, self.beam_width * self.beam_width
        )

        top_vals_per_input, top_idx_per_input = torch.topk(
            cand_scores_per_input, k=self.beam_width, dim=1
        )

        chosen_tokens = (
            topk_idxs.view(self.batch_size, self.beam_width * self.beam_width)
            .gather(1, top_idx_per_input)
            .view(-1)
        )

        chosen_beam_indices = (
            top_idx_per_input // self.beam_width
            + (
                torch.arange(self.batch_size, device=self.device) * self.beam_width
            ).unsqueeze(1)
        ).view(-1)

        self.sequences = torch.cat(
            [self.sequences[chosen_beam_indices], chosen_tokens.unsqueeze(1)], dim=1
        )

        self.finishes = self.finishes[chosen_beam_indices] | (
            chosen_tokens == self.end_token
        )
        self.scores = top_vals_per_input.view(-1).clone()

        return chosen_tokens, chosen_beam_indices

    def finalize_best_beams(self):
        scores_per_input = self.scores.view(self.batch_size, self.beam_width)
        finishes_per_input = self.finishes.view(self.batch_size, self.beam_width)

        best_beam_per_input = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        for i in range(self.batch_size):
            if finishes_per_input[i].any():
                tmp = scores_per_input[i].clone()
                tmp[~finishes_per_input[i]] = -1e9
                best_beam_per_input[i] = torch.argmax(tmp)
            else:
                best_beam_per_input[i] = torch.argmax(scores_per_input[i])

        return (
            best_beam_per_input
            + (torch.arange(self.batch_size, device=self.device) * self.beam_width)
        ).view(-1)
