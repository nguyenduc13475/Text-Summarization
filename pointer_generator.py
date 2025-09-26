import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from beam_search import beam_search


class PointerGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        encoder_hidden_dim=160,
        decoder_hidden_dim=196,
        attention_dim=224,
        bottle_neck_dim=56,
        unknown_token=3,
        start_token=0,
        end_token=1,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim, encoder_hidden_dim, batch_first=True, bidirectional=True
        )
        self.enc_to_dec_hidden = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.enc_to_dec_cell = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.decoder_cell = nn.LSTMCell(embedding_dim, decoder_hidden_dim)
        self.enc_hidden_to_attn = nn.Linear(encoder_hidden_dim * 2, attention_dim)
        self.dec_hidden_to_attn = nn.Linear(
            decoder_hidden_dim, attention_dim, bias=False
        )
        self.coverage_to_attn = nn.Linear(1, attention_dim, bias=False)
        self.attn_proj = nn.Linear(attention_dim, 1, bias=False)
        self.vocab_proj_1 = nn.Linear(
            decoder_hidden_dim + encoder_hidden_dim * 2, bottle_neck_dim
        )
        self.bottle_neck_activation = nn.Tanh()
        self.vocab_proj_2 = nn.Linear(bottle_neck_dim, vocab_size)
        self.context_to_switch = nn.Linear(encoder_hidden_dim * 2, 1)
        self.dec_hidden_to_switch = nn.Linear(decoder_hidden_dim, 1, bias=False)
        self.embedding_to_switch = nn.Linear(embedding_dim, 1, bias=False)

        self.vocab_size = vocab_size
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

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
        encoder_hidden_states, (
            encoder_final_hidden_state,
            encoder_final_cell_state,
        ) = self.encoder(embeddings)

        encoder_final_hidden_state = encoder_final_hidden_state.transpose(1, 0).reshape(
            encoder_final_hidden_state.shape[1], -1
        )
        encoder_final_cell_state = encoder_final_cell_state.transpose(1, 0).reshape(
            encoder_final_cell_state.shape[1], -1
        )

        decoder_hidden_state = self.enc_to_dec_hidden(encoder_final_hidden_state)
        decoder_cell_state = self.enc_to_dec_cell(encoder_final_cell_state)

        coverage_vector = torch.zeros(batch_size, max_sequence_length)
        current_token = torch.tensor([self.start_token] * batch_size)

        losses = torch.zeros(batch_size)
        num_tokens = 0

        for t in range(max_summary_length):
            current_embedding = self.embedding_layer(current_token)

            decoder_hidden_state, decoder_cell_state = self.decoder_cell(
                current_embedding, (decoder_hidden_state, decoder_cell_state)
            )
            attention_scores = F.tanh(
                self.enc_hidden_to_attn(encoder_hidden_states)
                + self.dec_hidden_to_attn(decoder_hidden_state).unsqueeze(1)
                + self.coverage_to_attn(coverage_vector.unsqueeze(2))
            )
            attention_scores = self.attn_proj(attention_scores).squeeze(2)
            attention_distribution = F.softmax(attention_scores, dim=1)
            context_vector = torch.bmm(
                attention_distribution.unsqueeze(1), encoder_hidden_states
            ).squeeze(1)
            hidden_context = torch.cat([decoder_hidden_state, context_vector], dim=1)
            vocab_distribution = F.softmax(
                self.vocab_proj_2(
                    self.bottle_neck_activation(self.vocab_proj_1(hidden_context))
                ),
                dim=1,
            )
            p_gen = F.sigmoid(
                self.context_to_switch(context_vector)
                + self.dec_hidden_to_switch(decoder_hidden_state)
                + self.embedding_to_switch(current_embedding)
            ).squeeze(1)

            next_token = labels[:, t]
            current_token = torch.where(
                next_token >= self.vocab_size,
                torch.tensor(self.unknown_token),
                next_token,
            )
            next_token_probs = p_gen * F.pad(vocab_distribution, (0, num_oovs)).gather(
                1, next_token.unsqueeze(1)
            ).squeeze(1) + (1 - p_gen) * (
                attention_distribution * (input_ids == next_token.unsqueeze(1)).float()
            ).sum(
                dim=1
            )

            nll_losses = -torch.log(next_token_probs + 1e-9)
            cov_losses = torch.min(attention_distribution, coverage_vector).sum(dim=1)
            if input_lengths is not None:
                nll_losses = nll_losses.masked_fill(input_lengths <= t, 0)
                cov_losses = cov_losses.masked_fill(input_lengths <= t, 0)
            losses = losses + nll_losses + cov_losses
            num_tokens += torch.sum(input_lengths > t).item()
            coverage_vector = coverage_vector + attention_distribution

        average_loss = losses.sum() / num_tokens

        self.optimizer.zero_grad()
        average_loss.backward()
        self.optimizer.step()
        return average_loss.item()

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
            encoder_final_cell_state,
        ) = self.encoder(embeddings)

        encoder_final_hidden_state = encoder_final_hidden_state.reshape(-1)
        encoder_final_cell_state = encoder_final_cell_state.reshape(-1)

        # state {decoder_hidden_state, decoder_cell_state, coverage_vector, sequence}
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
            attention_scores = F.tanh(
                self.enc_hidden_to_attn(encoder_hidden_states)
                + self.dec_hidden_to_attn(new_state["decoder_hidden_state"]).unsqueeze(
                    0
                )
                + self.coverage_to_attn(state["coverage_vector"].unsqueeze(1))
            )

            attention_scores = self.attn_proj(attention_scores).squeeze(1)
            attention_distribution = F.softmax(attention_scores, dim=0)
            context_vector = attention_distribution @ encoder_hidden_states
            hidden_context = torch.cat(
                [new_state["decoder_hidden_state"], context_vector], dim=0
            )
            vocab_distribution = F.softmax(
                self.vocab_proj_2(
                    self.bottle_neck_activation(self.vocab_proj_1(hidden_context))
                ),
                dim=0,
            )
            p_gen = F.sigmoid(
                self.context_to_switch(context_vector)
                + self.dec_hidden_to_switch(new_state["decoder_hidden_state"])
                + self.embedding_to_switch(current_embedding)
            ).squeeze(0)

            generator_probs = F.pad(p_gen * vocab_distribution, (0, num_oovs))
            pointer_probs = (1 - p_gen) * attention_distribution
            final_distribution = generator_probs.scatter_add(
                0, input_ids, pointer_probs
            )

            new_state["coverage_vector"] = (
                state["coverage_vector"] + attention_distribution
            )

            return final_distribution, new_state

        summary = beam_search(
            predictor=predictor,
            start_state={
                "decoder_hidden_state": self.enc_to_dec_hidden(
                    encoder_final_hidden_state
                ),
                "decoder_cell_state": self.enc_to_dec_cell(encoder_final_cell_state),
                "coverage_vector": torch.zeros(len(input_ids)),
                "sequence": [self.start_token],
            },
            beam_width=beam_width,
            max_state_length=max_summary_length,
            end_state_indicator=lambda state: state["sequence"][-1] == self.end_token,
        )["sequence"]

        return summary[1:]
