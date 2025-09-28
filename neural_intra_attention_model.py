import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from beam_search import beam_search
from dataset import token_ids_to_text
from metrics import compute_metric
from utils import tensor_dict_to_scalar


class NeuralIntraAttentionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        tokenizer,
        embedding_dim=128,
        hidden_dim=160,
        rl_loss_factor=0.75,
        learning_rate=1e-3,
        device="cpu",
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
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
        self.out_bias = nn.Parameter(torch.randn(vocab_size))
        self.concat_state_to_switch = nn.Linear(6 * hidden_dim, 1)

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.tokenizer = tokenizer

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.rl_loss_factor = rl_loss_factor

    def compute_loss(
        self, input_ids, labels, oov_lists, input_lengths=None, max_reinforce_length=100
    ):
        num_oovs = (
            max(input_ids.max().item(), self.vocab_size - 1) - self.vocab_size + 1
        )
        batch_size, max_sequence_length = input_ids.shape
        max_summary_length = labels.shape[1]

        embeddings = self.embedding_layer(
            torch.where(
                input_ids >= self.vocab_size,
                torch.tensor(self.unknown_token),
                input_ids,
            )
        )  # [8, 1826, 128]

        encoder_hidden_states, (
            encoder_final_hidden_state,
            _,
        ) = self.encoder(embeddings)
        out_proj = F.tanh(self.embedding_layer.weight @ self.vocab_proj)

        tf_loss = torch.zeros(batch_size)

        random_metrics = []
        greedy_metrics = []

        modes = [
            {
                "name": "teacher",
                "max_steps": max_summary_length,
            },
            {
                "name": "random",
                "max_steps": max_reinforce_length,
            },
            {
                "name": "greedy",
                "max_steps": max_reinforce_length,
            },
        ]

        for mode in modes:
            name = mode["name"]
            steps = mode["max_steps"]

            decoder_hidden_state = encoder_final_hidden_state.transpose(1, 0).reshape(
                encoder_final_hidden_state.shape[1], -1
            )
            decoder_cell_state = torch.zeros(batch_size, self.hidden_dim * 2)
            current_token = torch.tensor([self.start_token] * batch_size)
            cummulative_encoder_attention_scores = torch.zeros(
                batch_size, max_sequence_length
            )
            decoder_hidden_states = torch.empty(batch_size, 0, self.hidden_dim * 2)

            if name == "teacher":
                num_tokens = 0
            elif name == "random":
                cummulative_random_log_probs = torch.zeros(batch_size)
                is_continue = torch.ones(batch_size)
                random_sequences = torch.empty(batch_size, 0).long()
            elif name == "greedy":
                is_continue = torch.ones(batch_size)
                greedy_sequences = torch.empty(batch_size, 0).long()

            for t in range(steps):
                current_embedding = self.embedding_layer(current_token)
                decoder_hidden_state, decoder_cell_state = self.decoder_cell(
                    current_embedding, (decoder_hidden_state, decoder_cell_state)
                )

                encoder_attention_scores = (
                    (decoder_hidden_state @ self.encoder_attn_proj).unsqueeze(1)
                    @ encoder_hidden_states.transpose(1, 2)
                ).squeeze(1)

                if t == 0:
                    encoder_temporal_scores = encoder_attention_scores
                else:
                    encoder_temporal_scores = (
                        torch.exp(encoder_attention_scores)
                        / cummulative_encoder_attention_scores
                    )

                encoder_attention_distribution = F.softmax(
                    encoder_temporal_scores, dim=1
                )

                encoder_context_vector = torch.bmm(
                    encoder_attention_distribution.unsqueeze(1), encoder_hidden_states
                ).squeeze(1)

                if t == 0:
                    decoder_context_vector = torch.zeros(
                        batch_size, self.hidden_dim * 2
                    )
                else:
                    decoder_attention_scores = (
                        (decoder_hidden_state @ self.decoder_attn_proj).unsqueeze(1)
                        @ decoder_hidden_states.transpose(1, 2)
                    ).squeeze(1)

                    decoder_attention_distribution = F.softmax(
                        decoder_attention_scores, dim=1
                    )
                    decoder_context_vector = torch.bmm(
                        decoder_attention_distribution.unsqueeze(1),
                        decoder_hidden_states,
                    ).squeeze(1)

                concat_state = torch.cat(
                    [
                        decoder_hidden_state,
                        encoder_context_vector,
                        decoder_context_vector,
                    ],
                    dim=1,
                )

                vocab_distribution = F.softmax(
                    concat_state @ out_proj.T + self.out_bias, dim=1
                )

                if name == "teacher":
                    p_copy = F.sigmoid(
                        self.concat_state_to_switch(concat_state)
                    ).squeeze(1)
                    next_token = labels[:, t]
                    current_token = torch.where(
                        next_token >= self.vocab_size,
                        torch.tensor(self.unknown_token),
                        next_token,
                    )
                    next_token_probs = (1 - p_copy) * F.pad(
                        vocab_distribution, (0, num_oovs)
                    ).gather(1, next_token.unsqueeze(1)).squeeze(1) + p_copy * (
                        encoder_attention_distribution
                        * (input_ids == next_token.unsqueeze(1)).float()
                    ).sum(
                        dim=1
                    )

                    nll_loss = -torch.log(next_token_probs + 1e-9)
                    if input_lengths is not None:
                        nll_loss = nll_loss.masked_fill(input_lengths <= t, 0)
                    tf_loss = tf_loss + nll_loss
                    num_tokens += torch.sum(input_lengths > t).item()

                elif name == "random":
                    p_copy = F.sigmoid(self.concat_state_to_switch(concat_state))
                    generator_probs = F.pad(
                        (1 - p_copy) * vocab_distribution, (0, num_oovs)
                    )
                    pointer_probs = p_copy * encoder_attention_distribution
                    final_distribution = generator_probs.scatter_add(
                        1, input_ids, pointer_probs
                    )

                    next_token = torch.multinomial(
                        final_distribution, num_samples=1
                    ).squeeze(1)
                    next_token_probs = final_distribution[torch.arange(8), next_token]
                    next_token_log_probs = torch.log(next_token_probs + 1e-9)
                    cummulative_random_log_probs = (
                        cummulative_random_log_probs
                        + next_token_log_probs * is_continue
                    )
                    is_continue = is_continue * (next_token != self.end_token)
                    current_token = torch.where(
                        next_token >= self.vocab_size,
                        torch.tensor(self.unknown_token),
                        next_token,
                    )

                    random_seq_step = (
                        torch.where(is_continue.bool(), next_token, self.end_token)
                    ).unsqueeze(1)
                    random_sequences = torch.cat(
                        [random_sequences, random_seq_step], dim=1
                    )

                elif name == "greedy":
                    p_copy = F.sigmoid(self.concat_state_to_switch(concat_state))
                    generator_probs = F.pad(
                        (1 - p_copy) * vocab_distribution, (0, num_oovs)
                    )
                    pointer_probs = p_copy * encoder_attention_distribution
                    final_distribution = generator_probs.scatter_add(
                        1, input_ids, pointer_probs
                    )

                    next_token = torch.argmax(final_distribution, dim=1)
                    is_continue = is_continue * (next_token != self.end_token)
                    current_token = torch.where(
                        next_token >= self.vocab_size,
                        torch.tensor(self.unknown_token),
                        next_token,
                    )

                    greedy_seq_step = (
                        torch.where(is_continue.bool(), next_token, self.end_token)
                    ).unsqueeze(1)
                    greedy_sequences = torch.cat(
                        [greedy_sequences, greedy_seq_step], dim=1
                    )

                cummulative_encoder_attention_scores = (
                    cummulative_encoder_attention_scores + encoder_attention_scores
                )

                decoder_hidden_states = torch.cat(
                    [
                        decoder_hidden_states,
                        decoder_hidden_state.unsqueeze(1),
                    ],
                    dim=1,
                )

            if name == "teacher":
                tf_loss = tf_loss.sum() / num_tokens
            elif name == "random":
                random_metrics = []
                for b, random_sequence in enumerate(random_sequences):
                    random_metric = compute_metric(
                        "rouge2",
                        token_ids_to_text(
                            self.tokenizer,
                            random_sequence,
                            oov_lists[b],
                            self.vocab_size,
                        ),
                        token_ids_to_text(
                            self.tokenizer, labels[b], oov_lists[b], self.vocab_size
                        ),
                    )
                    random_metrics.append(random_metric)
            elif name == "greedy":
                greedy_metrics = []
                for b, greedy_sequence in enumerate(greedy_sequences):
                    greedy_metric = compute_metric(
                        "rouge2",
                        token_ids_to_text(
                            self.tokenizer,
                            greedy_sequence,
                            oov_lists[b],
                            self.vocab_size,
                        ),
                        token_ids_to_text(
                            self.tokenizer, labels[b], oov_lists[b], self.vocab_size
                        ),
                    )
                    greedy_metrics.append(greedy_metric)

        rl_loss = (
            (torch.tensor(greedy_metrics) - torch.tensor(random_metrics))
            * cummulative_random_log_probs
        ).sum()

        total_loss = tf_loss + rl_loss * self.rl_loss_factor

        return {"tf_loss": tf_loss, "rl_loss": rl_loss, "total_loss": total_loss}

    def train_one_batch(
        self, input_ids, labels, oov_lists, input_lengths=None, max_reinforce_length=100
    ):
        losses = self.compute_loss(
            input_ids, labels, oov_lists, input_lengths=None, max_reinforce_length=100
        )

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()
        return tensor_dict_to_scalar(losses)

    def validate_one_batch(
        self, input_ids, labels, oov_lists, input_lengths=None, max_reinforce_length=100
    ):
        with torch.no_grad():
            losses = self.compute_loss(
                input_ids,
                labels,
                oov_lists,
                input_lengths=None,
                max_reinforce_length=100,
            )
            return tensor_dict_to_scalar(losses)

    def infer(
        self,
        input_ids,
        max_summary_length=100,
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

            p_copy = F.sigmoid(self.concat_state_to_switch(concat_state)).squeeze(0)
            generator_probs = F.pad((1 - p_copy) * vocab_distribution, (0, num_oovs))
            pointer_probs = p_copy * encoder_attention_distribution
            final_distribution = generator_probs.scatter_add(
                0, input_ids, pointer_probs
            )

            # cuối
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
            max_state_length=max_summary_length,
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
            | ({"embedding": embeddings} if return_embedding else None)
        )
