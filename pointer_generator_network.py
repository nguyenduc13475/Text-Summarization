import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from beam_search import beam_search
from utils import tensor_dict_to_scalar


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


class PointerGeneratorNetwork(nn.Module):
    def __init__(
        self,
        tokenizer,
        embedding_dim=128,
        encoder_hidden_dim=160,
        decoder_hidden_dim=196,
        attention_dim=224,
        bottle_neck_dim=56,
        cov_loss_factor=0.75,
        learning_rate=1e-3,
        device="cpu",
    ):
        super().__init__()
        self.vocab_size = tokenizer.get_vocab_size()
        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_dim)
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
        self.context_to_switch = nn.Linear(encoder_hidden_dim * 2, 1)
        self.dec_hidden_to_switch = nn.Linear(decoder_hidden_dim, 1, bias=False)
        self.embedding_to_switch = nn.Linear(embedding_dim, 1, bias=False)

        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")
        self.vocab_proj_2 = nn.Linear(bottle_neck_dim, self.vocab_size - self.end_token)

        self.apply(init_weights)
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.cov_loss_factor = cov_loss_factor
        self.loss_scale = 1e-3

    def compute_loss(
        self, batch_input_ids, batch_target_ids, oov_lists, input_lengths=None
    ):
        max_num_oovs = 0
        for oov_list in oov_lists:
            if len(oov_list) > max_num_oovs:
                max_num_oovs = len(oov_list)

        batch_size, max_input_length = batch_input_ids.shape
        max_target_length = batch_target_ids.shape[1]

        batch_embeddings = self.embedding_layer(
            torch.where(
                batch_input_ids >= self.vocab_size,
                torch.tensor(self.unknown_token),
                batch_input_ids,
            )
        )
        batch_encoder_hidden_states, (
            encoder_final_hidden_states,
            encoder_final_cell_states,
        ) = self.encoder(batch_embeddings)

        encoder_final_hidden_states = encoder_final_hidden_states.transpose(
            1, 0
        ).reshape(batch_size, -1)
        encoder_final_cell_states = encoder_final_cell_states.transpose(1, 0).reshape(
            batch_size, -1
        )

        decoder_hidden_states = self.enc_to_dec_hidden(encoder_final_hidden_states)
        decoder_cell_states = self.enc_to_dec_cell(encoder_final_cell_states)

        coverage_vectors = torch.zeros(batch_size, max_input_length)
        current_tokens = torch.tensor([self.start_token] * batch_size)

        nll_losses = torch.zeros(batch_size)
        cov_losses = torch.zeros(batch_size)

        for t in range(max_target_length):
            current_embeddings = self.embedding_layer(current_tokens)

            decoder_hidden_states, decoder_cell_states = self.decoder_cell(
                current_embeddings, (decoder_hidden_states, decoder_cell_states)
            )
            batch_attention_scores = F.tanh(
                self.enc_hidden_to_attn(batch_encoder_hidden_states)
                + self.dec_hidden_to_attn(decoder_hidden_states).unsqueeze(1)
                + self.coverage_to_attn(coverage_vectors.unsqueeze(2))
            )
            batch_attention_scores = self.attn_proj(batch_attention_scores).squeeze(2)
            batch_attention_scores = batch_attention_scores.masked_fill(
                batch_input_ids == self.pad_token, float("-inf")
            )

            attention_distributions = F.softmax(batch_attention_scores, dim=1)
            context_vectors = torch.bmm(
                attention_distributions.unsqueeze(1), batch_encoder_hidden_states
            ).squeeze(1)
            hidden_contexts = torch.cat([decoder_hidden_states, context_vectors], dim=1)
            vocab_distributions = F.softmax(
                self.vocab_proj_2(
                    self.bottle_neck_activation(self.vocab_proj_1(hidden_contexts))
                ),
                dim=1,
            )
            p_gens = F.sigmoid(
                self.context_to_switch(context_vectors)
                + self.dec_hidden_to_switch(decoder_hidden_states)
                + self.embedding_to_switch(current_embeddings)
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
                attention_distributions
                * (batch_input_ids == next_tokens.unsqueeze(1)).float()
            ).sum(
                dim=1
            )

            nll_losses = nll_losses - torch.log(next_token_probs + 1e-9)
            cov_losses = cov_losses + torch.min(
                attention_distributions, coverage_vectors
            ).sum(dim=1)
            if input_lengths is not None:
                nll_losses = nll_losses.masked_fill(input_lengths <= t, 0)
                cov_losses = cov_losses.masked_fill(input_lengths <= t, 0)
            coverage_vectors = coverage_vectors + attention_distributions

        nll_loss = nll_losses.sum()
        cov_loss = cov_losses.sum()
        total_loss = nll_loss + cov_loss * self.cov_loss_factor

        return {"nll_loss": nll_loss, "cov_loss": cov_loss, "total_loss": total_loss}

    def train_one_batch(
        self, batch_input_ids, batch_target_ids, oov_lists, input_lengths=None
    ):
        losses = self.compute_loss(
            batch_input_ids, batch_target_ids, oov_lists, input_lengths
        )

        self.optimizer.zero_grad()
        (losses["total_loss"] * self.loss_scale).backward()
        self.optimizer.step()
        return tensor_dict_to_scalar(losses)

    def validate_one_batch(
        self, batch_input_ids, batch_target_ids, oov_lists, input_lengths=None
    ):
        with torch.no_grad():
            losses = self.compute_loss(
                batch_input_ids, batch_target_ids, oov_lists, input_lengths
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
            encoder_final_cell_state,
        ) = self.encoder(embeddings)

        encoder_final_hidden_state = encoder_final_hidden_state.reshape(-1)
        encoder_final_cell_state = encoder_final_cell_state.reshape(-1)

        def predictor(state):
            nonlocal return_attention
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

            generator_probs = F.pad(
                p_gen * vocab_distribution, (self.end_token, num_oovs)
            )
            pointer_probs = (1 - p_gen) * attention_distribution
            final_distribution = generator_probs.scatter_add(
                0, input_ids, pointer_probs
            )

            new_state["coverage_vector"] = (
                state["coverage_vector"] + attention_distribution
            )

            if return_attention:
                new_state["input_attention_distributions"] = state[
                    "input_attention_distributions"
                ] + [attention_distribution]

            return final_distribution, new_state

        beam_search_final_state = beam_search(
            predictor=predictor,
            start_state={
                "decoder_hidden_state": self.enc_to_dec_hidden(
                    encoder_final_hidden_state
                ),
                "decoder_cell_state": self.enc_to_dec_cell(encoder_final_cell_state),
                "coverage_vector": torch.zeros(len(input_ids)),
                "sequence": [self.start_token],
            }
            | (
                {
                    "input_attention_distributions": [],
                }
                if return_attention
                else {}
            ),
            beam_width=beam_width,
            max_state_length=max_output_length,
            end_state_indicator=lambda state: state["sequence"][-1] == self.end_token,
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
                }
                if return_attention
                else {}
            )
            | ({"embedding": embeddings} if return_embedding else {})
        )
