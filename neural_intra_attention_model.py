import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from beam_search import beam_search
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

    elif isinstance(m, nn.Parameter):
        if m.dim() > 1:
            init.xavier_uniform_(m.data)
        else:
            init.zeros_(m.data)


class NeuralIntraAttentionModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        embedding_dim=128,
        hidden_dim=160,
        rl_loss_factor=0.75,
        learning_rate=1e-3,
        device="cpu",
    ):
        super().__init__()
        self.vocab_size = tokenizer.get_vocab_size()
        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.decoder_cell = nn.LSTMCell(embedding_dim, hidden_dim * 2)
        self.encoder_attn_proj = nn.Parameter(
            torch.randn(hidden_dim * 2, hidden_dim * 2)
        )
        self.decoder_attn_proj = nn.Parameter(
            torch.randn(hidden_dim * 2, hidden_dim * 2)
        )
        self.vocab_proj = nn.Parameter(torch.randn(embedding_dim, 6 * hidden_dim))
        self.concat_state_to_switch = nn.Linear(6 * hidden_dim, 1)

        self.hidden_dim = hidden_dim
        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")
        self.tokenizer = tokenizer
        self.out_bias = nn.Parameter(torch.randn(self.vocab_size - self.end_token))

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.rl_loss_factor = rl_loss_factor
        self.loss_scale = 1e-3

    def compute_loss(
        self,
        batch_input_ids,
        batch_target_ids,
        oov_lists,
        input_lengths=None,
        max_reinforce_length=100,
        target_texts=None,
    ):
        max_num_oovs = 0
        for oov_list in oov_lists:
            if len(oov_list) > max_num_oovs:
                max_num_oovs = len(oov_list)

        batch_size, max_input_length = batch_input_ids.shape
        max_target_length = batch_target_ids.shape[1]
        out_proj = F.tanh(
            self.embedding_layer.weight[self.end_token :] @ self.vocab_proj
        )

        batch_embeddings = self.embedding_layer(
            torch.where(
                batch_input_ids >= self.vocab_size,
                torch.tensor(self.unknown_token),
                batch_input_ids,
            )
        )

        batch_encoder_hidden_states, (
            encoder_final_hidden_states,
            _,
        ) = self.encoder(batch_embeddings)

        nll_losses = torch.zeros(batch_size)

        sampling_sequence_metrics = []
        greedy_sequence_metrics = []

        modes = [
            {
                "name": "teacher",
                "max_steps": max_target_length,
            },
            {
                "name": "sampling",
                "max_steps": max_reinforce_length,
            },
            {
                "name": "greedy",
                "max_steps": max_reinforce_length,
            },
        ]

        for mode in modes:
            decoder_hidden_states = encoder_final_hidden_states.transpose(1, 0).reshape(
                encoder_final_hidden_states.shape[1], -1
            )
            decoder_cell_states = torch.zeros(batch_size, self.hidden_dim * 2)
            current_tokens = torch.tensor([self.start_token] * batch_size)
            batch_cummulative_encoder_attention_scores = torch.zeros(
                batch_size, max_input_length
            )
            previous_decoder_hidden_states = torch.empty(
                batch_size, 0, self.hidden_dim * 2
            )

            if mode["name"] in ["sampling", "greedy"]:
                is_continues = torch.ones(batch_size)

            if mode["name"] == "sampling":
                cummulative_sampling_log_probs = torch.zeros(batch_size)
                sampling_sequences = torch.empty(batch_size, 0).long()
            elif mode["name"] == "greedy":
                greedy_sequences = torch.empty(batch_size, 0).long()

            for t in range(mode["max_steps"]):
                current_embeddings = self.embedding_layer(current_tokens)
                decoder_hidden_states, decoder_cell_states = self.decoder_cell(
                    current_embeddings, (decoder_hidden_states, decoder_cell_states)
                )

                batch_encoder_attention_scores = (
                    (decoder_hidden_states @ self.encoder_attn_proj).unsqueeze(1)
                    @ batch_encoder_hidden_states.transpose(1, 2)
                ).squeeze(1)

                if t == 0:
                    batch_encoder_temporal_scores = batch_encoder_attention_scores
                else:
                    batch_encoder_temporal_scores = (
                        torch.exp(batch_encoder_attention_scores)
                        / batch_cummulative_encoder_attention_scores
                    )

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
                        batch_size, self.hidden_dim * 2
                    )
                else:
                    batch_decoder_attention_scores = (
                        (decoder_hidden_states @ self.decoder_attn_proj).unsqueeze(1)
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
                        decoder_hidden_states,
                        encoder_context_vectors,
                        decoder_context_vectors,
                    ],
                    dim=1,
                )

                vocab_distributions = F.softmax(
                    concat_states @ out_proj.T + self.out_bias, dim=1
                )

                if mode["name"] == "teacher":
                    p_gens = F.sigmoid(
                        self.concat_state_to_switch(concat_states)
                    ).squeeze(1)
                    next_tokens = batch_target_ids[:, t]
                    current_tokens = torch.where(
                        next_tokens >= self.vocab_size,
                        torch.tensor(self.unknown_token),
                        next_tokens,
                    )
                    next_token_probs = p_gens * F.pad(
                        vocab_distributions, (self.end_token, max_num_oovs)
                    ).gather(1, next_tokens.unsqueeze(1)).squeeze(1) + (1 - p_gens) * (
                        encoder_attention_distributions
                        * (batch_input_ids == next_tokens.unsqueeze(1)).float()
                    ).sum(
                        dim=1
                    )

                    nll_losses = nll_losses - (
                        torch.log(next_token_probs + 1e-9)
                        if input_lengths is None
                        else torch.log(next_token_probs + 1e-9).masked_fill(
                            input_lengths <= t, 0
                        )
                    )

                elif mode["name"] == "sampling":
                    p_gens = F.sigmoid(self.concat_state_to_switch(concat_states))
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
                        final_distributions[torch.arange(batch_size), next_tokens]
                        + 1e-9
                    )
                    cummulative_sampling_log_probs = (
                        cummulative_sampling_log_probs
                        + next_token_log_probs * is_continues
                    )
                    is_continues = is_continues * (next_tokens != self.end_token)
                    current_tokens = torch.where(
                        next_tokens >= self.vocab_size,
                        torch.tensor(self.unknown_token),
                        next_tokens,
                    )

                    sampling_sequence_steps = (
                        torch.where(is_continues.bool(), next_tokens, self.end_token)
                    ).unsqueeze(1)
                    sampling_sequences = torch.cat(
                        [sampling_sequences, sampling_sequence_steps], dim=1
                    )

                elif mode["name"] == "greedy":
                    p_gens = F.sigmoid(self.concat_state_to_switch(concat_states))
                    generator_probs = F.pad(
                        p_gens * vocab_distributions, (self.end_token, max_num_oovs)
                    )
                    pointer_probs = (1 - p_gens) * encoder_attention_distributions
                    final_distributions = generator_probs.scatter_add(
                        1, batch_input_ids, pointer_probs
                    )

                    next_tokens = torch.argmax(final_distributions, dim=1)
                    is_continues = is_continues * (next_tokens != self.end_token)
                    current_tokens = torch.where(
                        next_tokens >= self.vocab_size,
                        torch.tensor(self.unknown_token),
                        next_tokens,
                    )

                    greedy_sequence_steps = (
                        torch.where(is_continues.bool(), next_tokens, self.end_token)
                    ).unsqueeze(1)
                    greedy_sequences = torch.cat(
                        [greedy_sequences, greedy_sequence_steps], dim=1
                    )

                batch_cummulative_encoder_attention_scores = (
                    batch_cummulative_encoder_attention_scores
                    + batch_encoder_attention_scores
                )

                previous_decoder_hidden_states = torch.cat(
                    [
                        previous_decoder_hidden_states,
                        decoder_hidden_states.unsqueeze(1),
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
                            self.vocab_size,
                        ),
                        (
                            target_texts[batch_idx]
                            if target_texts is not None
                            else token_ids_to_text(
                                self.tokenizer,
                                batch_target_ids[batch_idx],
                                oov_lists[batch_idx],
                                self.vocab_size,
                            )
                        ),
                    )
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
                            self.vocab_size,
                        ),
                        (
                            target_texts[batch_idx]
                            if target_texts is not None
                            else token_ids_to_text(
                                self.tokenizer,
                                batch_target_ids[batch_idx],
                                oov_lists[batch_idx],
                                self.vocab_size,
                            )
                        ),
                    )
                    greedy_sequence_metrics.append(greedy_sequence_metric)

        rl_loss = (
            (
                torch.tensor(greedy_sequence_metrics)
                - torch.tensor(sampling_sequence_metrics)
            )
            * cummulative_sampling_log_probs
        ).sum()

        total_loss = nll_loss + rl_loss * self.rl_loss_factor

        return {"nll_loss": nll_loss, "rl_loss": rl_loss, "total_loss": total_loss}

    def train_one_batch(
        self,
        batch_input_ids,
        batch_target_ids,
        oov_lists,
        input_lengths=None,
        max_reinforce_length=100,
        target_texts=None,
    ):
        losses = self.compute_loss(
            batch_input_ids,
            batch_target_ids,
            oov_lists,
            input_lengths,
            max_reinforce_length,
            target_texts,
        )

        self.optimizer.zero_grad()
        (losses["total_loss"] * self.loss_scale).backward()
        self.optimizer.step()
        return tensor_dict_to_scalar(losses)

    def validate_one_batch(
        self,
        batch_input_ids,
        batch_target_ids,
        oov_lists,
        input_lengths=None,
        max_reinforce_length=100,
        target_texts=None,
    ):
        with torch.no_grad():
            losses = self.compute_loss(
                batch_input_ids,
                batch_target_ids,
                oov_lists,
                input_lengths,
                max_reinforce_length,
                target_texts,
            )
            return tensor_dict_to_scalar(losses)

    def infer(
        self,
        input_ids,
        max_output_length=100,
        beam_width=4,
        return_attention=False,
        return_embedding=False,
    ):
        num_oovs = (
            max(input_ids.max().item(), self.vocab_size - 1) - self.vocab_size + 1
        )
        embeddings = self.embedding_layer(
            torch.where(
                input_ids >= self.vocab_size,
                torch.tensor(self.unknown_token),
                input_ids,
            )
        )
        encoder_hidden_states, (
            encoder_final_hidden_state,
            _,
        ) = self.encoder(embeddings)

        out_proj = F.tanh(self.embedding_layer.weight @ self.vocab_proj)

        def predictor(state):
            nonlocal out_proj, return_attention
            new_state = dict()
            current_embedding = self.embedding_layer(
                torch.tensor(state["sequence"][-1])
            )
            new_state["decoder_hidden_state"], new_state["decoder_cell_state"] = (
                self.decoder_cell(
                    current_embedding,
                    (state["decoder_hidden_state"], state["decoder_cell_state"]),
                )
            )

            encoder_attention_scores = (
                new_state["decoder_hidden_state"]
                @ self.encoder_attn_proj
                @ encoder_hidden_states.T
            )
            if len(state["sequence"]) == 1:
                encoder_temporal_scores = encoder_attention_scores
            else:
                encoder_temporal_scores = (
                    torch.exp(encoder_attention_scores)
                    / state["cummulative_encoder_attention_scores"]
                )

            encoder_attention_distribution = F.softmax(encoder_temporal_scores, dim=0)
            encoder_context_vector = (
                encoder_attention_distribution @ encoder_hidden_states
            )

            if len(state["sequence"]) == 1:
                decoder_context_vector = torch.zeros(self.hidden_dim * 2)
            else:
                decoder_attention_scores = (
                    new_state["decoder_hidden_state"]
                    @ self.decoder_attn_proj
                    @ state["decoder_hidden_states"].T
                )

                decoder_attention_distribution = F.softmax(
                    decoder_attention_scores, dim=0
                )
                decoder_context_vector = (
                    decoder_attention_distribution @ state["decoder_hidden_states"]
                )

            concat_state = torch.cat(
                [
                    new_state["decoder_hidden_state"],
                    encoder_context_vector,
                    decoder_context_vector,
                ],
                dim=0,
            )

            vocab_distribution = F.softmax(
                out_proj @ concat_state + self.out_bias, dim=0
            )

            p_gen = F.sigmoid(self.concat_state_to_switch(concat_state)).squeeze(0)
            generator_probs = F.pad(p_gen * vocab_distribution, (0, num_oovs))
            pointer_probs = (1 - p_gen) * encoder_attention_distribution
            final_distribution = generator_probs.scatter_add(
                0, input_ids, pointer_probs
            )

            new_state["cummulative_encoder_attention_scores"] = (
                state["cummulative_encoder_attention_scores"] + encoder_attention_scores
            )

            new_state["decoder_hidden_states"] = torch.cat(
                [
                    state["decoder_hidden_states"],
                    new_state["decoder_hidden_state"].unsqueeze(0),
                ],
                dim=0,
            )

            if return_attention:
                new_state["input_attention_distributions"] = state[
                    "input_attention_distributions"
                ] + [encoder_attention_distribution]

                new_state["output_attention_distributions"] = state[
                    "output_attention_distributions"
                ] + [decoder_attention_distribution]

            return final_distribution, new_state

        beam_search_final_state = beam_search(
            predictor=predictor,
            start_state={
                "decoder_hidden_state": encoder_final_hidden_state.reshape(-1),
                "decoder_cell_state": torch.zeros(self.hidden_dim * 2),
                "cummulative_encoder_attention_scores": torch.zeros(len(input_ids)),
                "decoder_hidden_states": torch.empty(0, self.hidden_dim * 2),
                "sequence": [self.start_token],
            }
            | (
                {
                    "input_attention_distributions": [],
                    "output_attention_distributions": [],
                }
                if return_attention
                else {}
            ),
            beam_width=beam_width,
            max_state_length=max_output_length,
            end_state_indicator=lambda state: state["sequence"][-1] == self.end_token,
            unrepeated_trigram=True,
        )

        return (
            {
                "summary": beam_search_final_state["sequence"][1:],
            }
            | (
                {
                    "input_attention_distributions": (
                        beam_search_final_state["input_attention_distributions"]
                    ),
                    "output_attention_distributions": (
                        beam_search_final_state["output_attention_distributions"]
                    ),
                }
                if return_attention
                else {}
            )
            | ({"embedding": embeddings} if return_embedding else {})
        )
