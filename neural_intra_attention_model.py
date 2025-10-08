import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from beam_search import BeamSearch
from metrics import compute_metric
from utils import tensor_dict_to_scalar, token_ids_to_text


def init_weights(m):
    if isinstance(m, nn.Embedding):
        init.uniform_(m.weight, -0.1, 0.1)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.orthogonal_(param.data)
            elif "bias" in name:
                init.zeros_(param.data)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)


class NeuralIntraAttentionModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        rl_loss_factor=1.0,
        learning_rate=1e-2,
        device="cpu",
    ):
        super().__init__()
        self.vocab_size = tokenizer.get_vocab_size()
        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim * 2,
            batch_first=True,
            num_layers=num_layers,
        )
        self.encoder_attn_proj = nn.Parameter(
            torch.empty(hidden_dim * 2, hidden_dim * 2)
        )
        torch.nn.init.xavier_uniform_(self.encoder_attn_proj)

        self.decoder_attn_proj = nn.Parameter(
            torch.empty(hidden_dim * 2, hidden_dim * 2)
        )
        torch.nn.init.xavier_uniform_(self.decoder_attn_proj)

        self.vocab_proj = nn.Parameter(torch.empty(embedding_dim, 6 * hidden_dim))
        torch.nn.init.xavier_uniform_(self.vocab_proj)

        self.concat_state_to_switch = nn.Linear(6 * hidden_dim, 1)

        self.hidden_dim = hidden_dim
        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")
        self.tokenizer = tokenizer
        self.out_bias = nn.Parameter(torch.zeros(self.vocab_size - self.end_token))
        self.device = torch.device(device)

        self.apply(init_weights)
        self.to(device)
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        self.rl_loss_factor = rl_loss_factor
        self.loss_scale = 1e-3

    def _safe_ids(self, ids):
        return torch.where(
            ids >= self.vocab_size,
            torch.tensor(self.unknown_token, device=self.device),
            ids,
        )

    def _safe_embed(self, ids):
        return self.embedding_layer(self._safe_ids(ids))

    def compute_loss(
        self,
        batch_input_ids,
        batch_target_ids,
        oov_lists,
        input_lengths,
        target_lengths,
        max_reinforce_length=100,
        target_texts=None,
    ):
        batch_input_ids = batch_input_ids.to(self.device)
        batch_target_ids = batch_target_ids.to(self.device)
        target_lengths = target_lengths.to(self.device)
        batch_size, max_input_length = batch_input_ids.shape
        max_target_length = batch_target_ids.shape[1]

        max_num_oovs = 0
        for oov_list in oov_lists:
            if len(oov_list) > max_num_oovs:
                max_num_oovs = len(oov_list)

        out_proj = torch.tanh(
            self.embedding_layer.weight[self.end_token :] @ self.vocab_proj
        )

        batch_embeddings = self._safe_embed(batch_input_ids)

        batch_encoder_hidden_states, (
            encoder_final_hidden_states,
            _,
        ) = self.encoder(
            pack_padded_sequence(
                batch_embeddings,
                input_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
        )
        batch_encoder_hidden_states, _ = pad_packed_sequence(
            batch_encoder_hidden_states, batch_first=True
        )

        nll_losses = torch.zeros(batch_size, device=self.device)

        sampling_sequence_metrics = []
        greedy_sequence_metrics = []

        modes = [
            {
                "name": "teacher",
                "max_steps": max_target_length,
            },
            # {
            #     "name": "sampling",
            #     "max_steps": max_reinforce_length,
            # },
            # {
            #     "name": "greedy",
            #     "max_steps": max_reinforce_length,
            # },
        ]

        for mode in modes:
            decoder_hidden_states = (
                encoder_final_hidden_states.reshape(self.num_layers, 2, batch_size, -1)
                .transpose(1, 2)
                .reshape(self.num_layers, batch_size, -1)
            )
            decoder_cell_states = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim * 2, device=self.device
            )
            current_tokens = torch.tensor(
                [self.start_token] * batch_size, device=self.device
            )
            batch_cummulative_encoder_attention_scores = torch.zeros(
                batch_size, max_input_length, device=self.device
            )
            previous_decoder_hidden_states = torch.empty(
                batch_size, 0, self.hidden_dim * 2, device=self.device
            )

            if mode["name"] in ["sampling", "greedy"]:
                is_continues = torch.ones(batch_size, device=self.device)

            if mode["name"] == "sampling":
                cummulative_sampling_log_probs = torch.zeros(
                    batch_size, device=self.device
                )
                sampling_sequences = torch.empty(
                    batch_size, 0, device=self.device, dtype=torch.long
                )
            elif mode["name"] == "greedy":
                greedy_sequences = torch.empty(
                    batch_size, 0, device=self.device, dtype=torch.long
                )

            for t in range(mode["max_steps"]):
                current_embeddings = self.embedding_layer(current_tokens)
                _, (decoder_hidden_states, decoder_cell_states) = self.decoder(
                    current_embeddings.unsqueeze(1),
                    (decoder_hidden_states, decoder_cell_states),
                )

                batch_encoder_attention_scores = (
                    (
                        (decoder_hidden_states[-1] @ self.encoder_attn_proj).unsqueeze(
                            1
                        )
                        @ batch_encoder_hidden_states.transpose(1, 2)
                    )
                    .squeeze(1)
                    .clamp(max=30)
                )

                if t == 0:
                    batch_encoder_temporal_scores = torch.exp(
                        batch_encoder_attention_scores
                    )
                else:
                    batch_encoder_temporal_scores = torch.exp(
                        batch_encoder_attention_scores
                    ) / (batch_cummulative_encoder_attention_scores + 1e-8)

                batch_encoder_temporal_scores = (
                    batch_encoder_temporal_scores.masked_fill(
                        batch_input_ids == self.pad_token, float("-inf")
                    )
                )
                encoder_attention_distributions = F.softmax(
                    batch_encoder_temporal_scores, dim=1
                )

                encoder_context_vectors = torch.bmm(
                    encoder_attention_distributions.unsqueeze(1),
                    batch_encoder_hidden_states,
                ).squeeze(1)

                if t == 0:
                    decoder_context_vectors = torch.zeros(
                        batch_size, self.hidden_dim * 2, device=self.device
                    )
                else:
                    batch_decoder_attention_scores = (
                        (decoder_hidden_states[-1] @ self.decoder_attn_proj).unsqueeze(
                            1
                        )
                        @ previous_decoder_hidden_states.transpose(1, 2)
                    ).squeeze(1)

                    decoder_attention_distributions = F.softmax(
                        batch_decoder_attention_scores, dim=1
                    )
                    decoder_context_vectors = torch.bmm(
                        decoder_attention_distributions.unsqueeze(1),
                        previous_decoder_hidden_states,
                    ).squeeze(1)

                concat_states = torch.cat(
                    [
                        decoder_hidden_states[-1],
                        encoder_context_vectors,
                        decoder_context_vectors,
                    ],
                    dim=1,
                )

                vocab_distributions = F.softmax(
                    concat_states @ out_proj.T + self.out_bias, dim=1
                )

                if mode["name"] == "teacher":
                    p_gens = torch.sigmoid(
                        self.concat_state_to_switch(concat_states)
                    ).squeeze(1)
                    next_tokens = batch_target_ids[:, t]
                    current_tokens = self._safe_ids(next_tokens)
                    next_token_probs = p_gens * F.pad(
                        vocab_distributions, (self.end_token, max_num_oovs)
                    ).gather(1, next_tokens.unsqueeze(1)).squeeze(1) + (1 - p_gens) * (
                        encoder_attention_distributions
                        * (batch_input_ids == next_tokens.unsqueeze(1)).float()
                    ).sum(
                        dim=1
                    )

                    nll_losses = nll_losses - (
                        torch.log(next_token_probs + 1e-9).masked_fill(
                            t >= target_lengths, 0
                        )
                    )

                elif mode["name"] == "sampling":
                    p_gens = torch.sigmoid(self.concat_state_to_switch(concat_states))
                    generator_probs = F.pad(
                        p_gens * vocab_distributions, (self.end_token, max_num_oovs)
                    )
                    pointer_probs = (1 - p_gens) * encoder_attention_distributions
                    final_distributions = generator_probs.scatter_add(
                        1, batch_input_ids, pointer_probs
                    )

                    next_tokens = torch.multinomial(
                        final_distributions, num_samples=1
                    ).squeeze(1)
                    next_token_log_probs = torch.log(
                        final_distributions[
                            torch.arange(batch_size, device=self.device), next_tokens
                        ]
                        + 1e-9
                    )
                    cummulative_sampling_log_probs = (
                        cummulative_sampling_log_probs
                        + next_token_log_probs * is_continues
                    )
                    is_continues = is_continues * (next_tokens != self.end_token)
                    current_tokens = self._safe_ids(next_tokens)

                    sampling_sequence_steps = (
                        torch.where(is_continues.bool(), next_tokens, self.end_token)
                    ).unsqueeze(1)
                    sampling_sequences = torch.cat(
                        [sampling_sequences, sampling_sequence_steps], dim=1
                    )

                elif mode["name"] == "greedy":
                    p_gens = torch.sigmoid(self.concat_state_to_switch(concat_states))
                    generator_probs = F.pad(
                        p_gens * vocab_distributions, (self.end_token, max_num_oovs)
                    )
                    pointer_probs = (1 - p_gens) * encoder_attention_distributions
                    final_distributions = generator_probs.scatter_add(
                        1, batch_input_ids, pointer_probs
                    )

                    next_tokens = torch.argmax(final_distributions, dim=1)
                    is_continues = is_continues * (next_tokens != self.end_token)
                    current_tokens = self._safe_ids(next_tokens)

                    greedy_sequence_steps = (
                        torch.where(is_continues.bool(), next_tokens, self.end_token)
                    ).unsqueeze(1)
                    greedy_sequences = torch.cat(
                        [greedy_sequences, greedy_sequence_steps], dim=1
                    )

                batch_cummulative_encoder_attention_scores = (
                    batch_cummulative_encoder_attention_scores
                    + torch.exp(batch_encoder_attention_scores)
                )

                previous_decoder_hidden_states = torch.cat(
                    [
                        previous_decoder_hidden_states,
                        decoder_hidden_states[-1].unsqueeze(1),
                    ],
                    dim=1,
                )

            if mode["name"] == "teacher":
                nll_loss = nll_losses.sum()
            elif mode["name"] == "sampling":
                sampling_sequence_metrics = []
                for batch_idx, sampling_sequence in enumerate(sampling_sequences):
                    sampling_sequence_metric = compute_metric(
                        "rouge2",
                        token_ids_to_text(
                            self.tokenizer,
                            sampling_sequence,
                            oov_lists[batch_idx],
                        ),
                        (
                            target_texts[batch_idx]
                            if target_texts is not None
                            else token_ids_to_text(
                                self.tokenizer,
                                batch_target_ids[batch_idx],
                                oov_lists[batch_idx],
                            )
                        ),
                    )["rouge2"][0]
                    sampling_sequence_metrics.append(sampling_sequence_metric)
            elif mode["name"] == "greedy":
                greedy_sequence_metrics = []
                for batch_idx, greedy_sequence in enumerate(greedy_sequences):
                    greedy_sequence_metric = compute_metric(
                        "rouge2",
                        token_ids_to_text(
                            self.tokenizer,
                            greedy_sequence,
                            oov_lists[batch_idx],
                        ),
                        (
                            target_texts[batch_idx]
                            if target_texts is not None
                            else token_ids_to_text(
                                self.tokenizer,
                                batch_target_ids[batch_idx],
                                oov_lists[batch_idx],
                            )
                        ),
                    )["rouge2"][0]
                    greedy_sequence_metrics.append(greedy_sequence_metric)

        rl_loss = 0
        # (
        #     (
        #         torch.tensor(greedy_sequence_metrics, device=self.device)
        #         - torch.tensor(sampling_sequence_metrics, device=self.device)
        #     )
        #     * cummulative_sampling_log_probs
        # ).sum()

        total_loss = nll_loss + rl_loss * self.rl_loss_factor

        return {"nll_loss": nll_loss, "rl_loss": rl_loss, "total_loss": total_loss}

    def train_one_batch(
        self,
        batch_input_ids,
        batch_target_ids,
        oov_lists,
        input_lengths,
        target_lengths,
        max_reinforce_length=100,
        target_texts=None,
    ):
        self.train()
        self.optimizer.zero_grad()

        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                losses = self.compute_loss(
                    batch_input_ids,
                    batch_target_ids,
                    oov_lists,
                    input_lengths,
                    target_lengths,
                    max_reinforce_length,
                    target_texts,
                )
            self.scaler.scale(losses["total_loss"] * self.loss_scale).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses = self.compute_loss(
                batch_input_ids,
                batch_target_ids,
                oov_lists,
                input_lengths,
                target_lengths,
                max_reinforce_length,
                target_texts,
            )
            (losses["total_loss"] * self.loss_scale).backward()
            self.optimizer.step()

        return tensor_dict_to_scalar(losses)

    def validate_one_batch(
        self,
        batch_input_ids,
        batch_target_ids,
        oov_lists,
        input_lengths,
        target_lengths,
        max_reinforce_length=100,
        target_texts=None,
    ):
        self.eval()
        with torch.no_grad():
            losses = self.compute_loss(
                batch_input_ids,
                batch_target_ids,
                oov_lists,
                input_lengths,
                target_lengths,
                max_reinforce_length,
                target_texts,
            )
            return tensor_dict_to_scalar(losses)

    def infer(
        self,
        batch_input_ids,
        max_output_length=100,
        beam_width=4,
        return_attention=False,
        return_embedding=False,
    ):
        self.eval()
        with torch.no_grad():
            if not isinstance(batch_input_ids, torch.Tensor):
                batch_input_ids = torch.tensor(
                    batch_input_ids, device=self.device, dtype=torch.long
                )
            batch_input_ids = batch_input_ids.to(self.device)
            beam_width = min(beam_width, self.vocab_size)
            if batch_input_ids.dim() == 1:
                batch_input_ids = batch_input_ids.unsqueeze(0)
            batch_size = batch_input_ids.shape[0]
            beam_search = BeamSearch(
                batch_size, beam_width, self.start_token, self.end_token, self.device
            )
            num_oovs = (
                max(batch_input_ids.max().item(), self.vocab_size - 1)
                - self.vocab_size
                + 1
            )

            out_proj = torch.tanh(
                self.embedding_layer.weight[self.end_token :] @ self.vocab_proj
            )

            batch_embeddings = self._safe_embed(batch_input_ids)

            batch_encoder_hidden_states, (
                encoder_final_hidden_states,
                _,
            ) = self.encoder(
                pack_padded_sequence(
                    batch_embeddings,
                    (batch_input_ids != self.pad_token).sum(dim=1).cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
            )

            batch_encoder_hidden_states, _ = pad_packed_sequence(
                batch_encoder_hidden_states, batch_first=True
            )

            decoder_hidden_states = (
                encoder_final_hidden_states.reshape(self.num_layers, 2, batch_size, -1)
                .transpose(1, 2)
                .reshape(self.num_layers, batch_size, -1)
            )
            decoder_cell_states = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim * 2, device=self.device
            )

            current_tokens = torch.tensor(
                [self.start_token] * batch_size, device=self.device
            )

            current_embeddings = self.embedding_layer(current_tokens)
            _, (decoder_hidden_states, decoder_cell_states) = self.decoder(
                current_embeddings.unsqueeze(1),
                (decoder_hidden_states, decoder_cell_states),
            )
            batch_encoder_temporal_scores = (
                (
                    (decoder_hidden_states[-1] @ self.encoder_attn_proj).unsqueeze(1)
                    @ batch_encoder_hidden_states.transpose(1, 2)
                )
                .squeeze(1)
                .clamp(max=30)
            )
            input_padding_mask = batch_input_ids == self.pad_token
            batch_encoder_temporal_scores = batch_encoder_temporal_scores.masked_fill(
                input_padding_mask, float("-inf")
            )
            encoder_attention_distributions = F.softmax(
                batch_encoder_temporal_scores, dim=1
            )

            encoder_context_vectors = torch.bmm(
                encoder_attention_distributions.unsqueeze(1),
                batch_encoder_hidden_states,
            ).squeeze(1)

            decoder_context_vectors = torch.zeros(
                batch_size, self.hidden_dim * 2, device=self.device
            )

            concat_states = torch.cat(
                [
                    decoder_hidden_states[-1],
                    encoder_context_vectors,
                    decoder_context_vectors,
                ],
                dim=1,
            )

            vocab_distributions = F.softmax(
                concat_states @ out_proj.T + self.out_bias, dim=1
            )

            p_gens = torch.sigmoid(self.concat_state_to_switch(concat_states))
            batch_generator_probs = F.pad(
                p_gens * vocab_distributions, (self.end_token, num_oovs)
            )
            batch_pointer_probs = (1 - p_gens) * encoder_attention_distributions
            batch_final_distributions = batch_generator_probs.scatter_add(
                1, batch_input_ids, batch_pointer_probs
            )

            chosen_tokens = beam_search.init_from_first_topk(batch_final_distributions)
            current_embeddings = self._safe_embed(chosen_tokens)
            decoder_hidden_states = decoder_hidden_states.repeat_interleave(
                beam_width, dim=1
            )
            decoder_cell_states = decoder_cell_states.repeat_interleave(
                beam_width, dim=1
            )
            batch_encoder_hidden_states = batch_encoder_hidden_states.repeat_interleave(
                beam_width, dim=0
            )
            batch_cummulative_encoder_attention_scores = (
                batch_encoder_temporal_scores.repeat_interleave(beam_width, dim=0)
            )
            input_padding_mask = input_padding_mask.repeat_interleave(beam_width, dim=0)
            previous_decoder_hidden_states = decoder_hidden_states[-1].unsqueeze(1)

            if return_attention:
                cross_attention_distributions_list = [
                    encoder_attention_distributions.repeat_interleave(beam_width, dim=0)
                ]
                decoder_attention_distributions_list = []

            batch_input_ids = batch_input_ids.repeat_interleave(beam_width, dim=0)

            for _ in range(2, max_output_length + 1):
                _, (decoder_hidden_states, decoder_cell_states) = self.decoder(
                    current_embeddings.unsqueeze(1),
                    (decoder_hidden_states, decoder_cell_states),
                )

                batch_encoder_attention_scores = (
                    (
                        (decoder_hidden_states[-1] @ self.encoder_attn_proj).unsqueeze(
                            1
                        )
                        @ batch_encoder_hidden_states.transpose(1, 2)
                    )
                    .squeeze(1)
                    .clamp(max=30)
                )

                batch_encoder_temporal_scores = (
                    torch.exp(batch_encoder_attention_scores)
                    / batch_cummulative_encoder_attention_scores
                )

                batch_encoder_temporal_scores = (
                    batch_encoder_temporal_scores.masked_fill(
                        input_padding_mask, float("-inf")
                    )
                )

                encoder_attention_distributions = F.softmax(
                    batch_encoder_temporal_scores, dim=1
                )

                encoder_context_vectors = torch.bmm(
                    encoder_attention_distributions.unsqueeze(1),
                    batch_encoder_hidden_states,
                ).squeeze(1)

                batch_cummulative_encoder_attention_scores = (
                    batch_cummulative_encoder_attention_scores
                    + torch.exp(batch_encoder_attention_scores)
                )

                batch_decoder_attention_scores = (
                    (decoder_hidden_states[-1] @ self.decoder_attn_proj).unsqueeze(1)
                    @ previous_decoder_hidden_states.transpose(1, 2)
                ).squeeze(1)

                decoder_attention_distributions = F.softmax(
                    batch_decoder_attention_scores, dim=1
                )
                decoder_context_vectors = torch.bmm(
                    decoder_attention_distributions.unsqueeze(1),
                    previous_decoder_hidden_states,
                ).squeeze(1)

                previous_decoder_hidden_states = torch.cat(
                    [
                        previous_decoder_hidden_states,
                        decoder_hidden_states[-1].unsqueeze(1),
                    ],
                    dim=1,
                )

                concat_states = torch.cat(
                    [
                        decoder_hidden_states[-1],
                        encoder_context_vectors,
                        decoder_context_vectors,
                    ],
                    dim=1,
                )

                vocab_distributions = F.softmax(
                    concat_states @ out_proj.T + self.out_bias, dim=1
                )

                p_gens = torch.sigmoid(self.concat_state_to_switch(concat_states))
                batch_generator_probs = F.pad(
                    p_gens * vocab_distributions, (self.end_token, num_oovs)
                )
                batch_pointer_probs = (1 - p_gens) * encoder_attention_distributions
                batch_final_distributions = batch_generator_probs.scatter_add(
                    1, batch_input_ids, batch_pointer_probs
                )

                chosen_tokens, chosen_beam_indices = beam_search.advance(
                    batch_final_distributions
                )
                current_embeddings = self._safe_embed(chosen_tokens)
                batch_input_ids = batch_input_ids[chosen_beam_indices]

                if return_attention:
                    cross_attention_distributions_list.append(
                        encoder_attention_distributions
                    )
                    for i in range(len(cross_attention_distributions_list)):
                        cross_attention_distributions_list[i] = (
                            cross_attention_distributions_list[i][chosen_beam_indices]
                        )

                    decoder_attention_distributions_list.append(
                        decoder_attention_distributions
                    )
                    for i in range(len(decoder_attention_distributions_list)):
                        decoder_attention_distributions_list[i] = (
                            decoder_attention_distributions_list[i][chosen_beam_indices]
                        )

                if beam_search.finishes.all():
                    break

            chosen_beam_indices = beam_search.finalize_best_beams()

            output = {"output_ids": beam_search.sequences[chosen_beam_indices, 1:]}
            if return_embedding:
                output["input_embeddings"] = batch_embeddings
            if return_attention:
                output["cross_attention_distributions"] = torch.stack(
                    [
                        attention_distributions[chosen_beam_indices]
                        for attention_distributions in cross_attention_distributions_list
                    ],
                    dim=1,
                )
                output["decoder_attention_distributions"] = torch.stack(
                    [
                        torch.nn.functional.pad(
                            attention_distributions[chosen_beam_indices],
                            (
                                0,
                                len(decoder_attention_distributions_list)
                                - attention_distributions[chosen_beam_indices].shape[1],
                            ),
                        )
                        for attention_distributions in decoder_attention_distributions_list
                    ],
                    dim=1,
                )

            return output
