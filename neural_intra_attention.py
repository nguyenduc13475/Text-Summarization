import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from beam_search import beam_search


class NeuralIntraAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=160,
        unknown_token=3,
        start_token=0,
        end_token=1,
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
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token

    def train_one_batch(self, input_ids, labels, input_lengths=None):
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

    def infer(self, input_ids, max_summary_length=100, beam_width=4):
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

        def predictor(state):
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

            out_proj = F.tanh(self.embedding_layer.weight @ self.vocab_proj)
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

            return final_distribution, new_state

        summary = beam_search(
            predictor=predictor,
            start_state={
                "decoder_hidden_state": encoder_final_hidden_state.reshape(-1),
                "decoder_cell_state": torch.zeros(self.hidden_dim * 2),
                "cummulative_encoder_attention_scores": torch.zeros(len(input_ids)),
                "decoder_hidden_states": torch.empty(0, self.hidden_dim * 2),
                "sequence": [self.start_token],
            },
            beam_width=beam_width,
            max_state_length=max_summary_length,
            end_state_indicator=lambda state: state["sequence"][-1] == self.end_token,
        )["sequence"]

        return summary[1:]
