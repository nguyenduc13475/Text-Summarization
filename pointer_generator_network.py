import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from beam_search import BeamSearch
from utils import tensor_dict_to_scalar


def init_weights(m):
    if isinstance(m, nn.Embedding):
        init.normal_(m.weight, mean=0.0, std=1.0)

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
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        attention_dim=256,
        bottle_neck_dim=512,
        num_layers=2,
        cov_loss_factor=1.0,
        learning_rate=1e-3,
        device="cpu",
    ):
        super().__init__()
        self.vocab_size = tokenizer.get_vocab_size()
        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim,
            encoder_hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )
        self.enc_to_dec_hidden = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.enc_to_dec_cell = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.decoder = nn.LSTM(
            embedding_dim,
            decoder_hidden_dim,
            batch_first=True,
            num_layers=num_layers,
        )
        self.enc_hidden_to_attn = nn.Linear(encoder_hidden_dim * 2, attention_dim)
        self.dec_hidden_to_attn = nn.Linear(
            decoder_hidden_dim, attention_dim, bias=False
        )
        self.coverage_to_attn = nn.Linear(1, attention_dim, bias=False)
        self.attn_proj = nn.Linear(attention_dim, 1, bias=False)
        self.vocab_proj_1 = nn.Linear(
            decoder_hidden_dim + encoder_hidden_dim * 2, bottle_neck_dim
        )
        self.bottle_neck_activation = nn.ReLU()
        self.context_to_switch = nn.Linear(encoder_hidden_dim * 2, 1)
        self.dec_hidden_to_switch = nn.Linear(decoder_hidden_dim, 1, bias=False)
        self.embedding_to_switch = nn.Linear(embedding_dim, 1, bias=False)

        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")
        self.vocab_proj_2 = nn.Linear(bottle_neck_dim, self.vocab_size - self.end_token)
        self.device = torch.device(device)
        self.apply(init_weights)
        self.to(device)
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        self.cov_loss_factor = cov_loss_factor
        self.loss_scale = 1e-2

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

        batch_embeddings = self._safe_embed(batch_input_ids)
        batch_encoder_hidden_states, (
            encoder_final_hidden_states,
            encoder_final_cell_states,
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

        encoder_final_hidden_states = (
            encoder_final_hidden_states.reshape(self.num_layers, 2, batch_size, -1)
            .transpose(1, 2)
            .reshape(self.num_layers, batch_size, -1)
        )
        encoder_final_cell_states = (
            encoder_final_cell_states.reshape(self.num_layers, 2, batch_size, -1)
            .transpose(1, 2)
            .reshape(self.num_layers, batch_size, -1)
        )

        decoder_hidden_states = self.enc_to_dec_hidden(encoder_final_hidden_states)
        decoder_cell_states = self.enc_to_dec_cell(encoder_final_cell_states)

        coverage_vectors = torch.zeros(batch_size, max_input_length, device=self.device)
        current_tokens = torch.tensor(
            [self.start_token] * batch_size, device=self.device
        )

        nll_losses = torch.zeros(batch_size, device=self.device)
        cov_losses = torch.zeros(batch_size, device=self.device)

        for t in range(max_target_length):
            current_embeddings = self.embedding_layer(current_tokens)

            _, (decoder_hidden_states, decoder_cell_states) = self.decoder(
                current_embeddings.unsqueeze(1),
                (decoder_hidden_states, decoder_cell_states),
            )
            batch_attention_scores = torch.tanh(
                self.enc_hidden_to_attn(batch_encoder_hidden_states)
                + self.dec_hidden_to_attn(decoder_hidden_states[-1]).unsqueeze(1)
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
            hidden_contexts = torch.cat(
                [decoder_hidden_states[-1], context_vectors], dim=1
            )
            vocab_distributions = F.softmax(
                self.vocab_proj_2(
                    self.bottle_neck_activation(self.vocab_proj_1(hidden_contexts))
                ),
                dim=1,
            )
            p_gens = torch.sigmoid(
                self.context_to_switch(context_vectors)
                + self.dec_hidden_to_switch(decoder_hidden_states[-1])
                + self.embedding_to_switch(current_embeddings)
            ).squeeze(1)

            next_tokens = batch_target_ids[:, t]
            current_tokens = self._safe_ids(next_tokens)
            next_token_probs = p_gens * F.pad(
                vocab_distributions, (self.end_token, max_num_oovs)
            ).gather(1, next_tokens.unsqueeze(1)).squeeze(1) + (1 - p_gens) * (
                attention_distributions
                * (batch_input_ids == next_tokens.unsqueeze(1)).float()
            ).sum(
                dim=1
            )

            nll_losses = nll_losses - torch.log(next_token_probs + 1e-9).masked_fill(
                t >= target_lengths, 0
            )
            cov_losses = cov_losses + torch.min(
                attention_distributions, coverage_vectors
            ).sum(dim=1).masked_fill(t >= target_lengths, 0)
            coverage_vectors = coverage_vectors + attention_distributions

        nll_loss = nll_losses.sum()
        cov_loss = cov_losses.sum()
        total_loss = nll_loss + cov_loss * self.cov_loss_factor

        return {"nll_loss": nll_loss, "cov_loss": cov_loss, "total_loss": total_loss}

    def train_one_batch(
        self,
        batch_input_ids,
        batch_target_ids,
        oov_lists,
        input_lengths,
        target_lengths,
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
                )
            self.scaler.scale(losses["total_loss"] * self.loss_scale).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses = self.compute_loss(
                batch_input_ids,
                batch_target_ids,
                oov_lists,
                input_lengths,
                target_lengths,
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
    ):
        self.eval()
        with torch.no_grad():
            losses = self.compute_loss(
                batch_input_ids,
                batch_target_ids,
                oov_lists,
                input_lengths,
                target_lengths,
            )
            return tensor_dict_to_scalar(losses)

    def infer(
        self,
        batch_input_ids,
        max_output_length=100,
        beam_width=6,
        trigram_penalty=-30,
        bigram_penalty=-15,
        unigram_penalty=-2,
        penalty_range=15,
        return_attention=False,
        return_embedding=False,
        **kwargs
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

            batch_embeddings = self._safe_embed(batch_input_ids)
            batch_encoder_hidden_states, (
                encoder_final_hidden_states,
                encoder_final_cell_states,
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

            encoder_final_hidden_states = (
                encoder_final_hidden_states.reshape(self.num_layers, 2, batch_size, -1)
                .transpose(1, 2)
                .reshape(self.num_layers, batch_size, -1)
            )
            encoder_final_cell_states = (
                encoder_final_cell_states.reshape(self.num_layers, 2, batch_size, -1)
                .transpose(1, 2)
                .reshape(self.num_layers, batch_size, -1)
            )

            decoder_hidden_states = self.enc_to_dec_hidden(encoder_final_hidden_states)
            decoder_cell_states = self.enc_to_dec_cell(encoder_final_cell_states)

            current_tokens = torch.tensor(
                [self.start_token] * batch_size, device=self.device
            )

            current_embeddings = self.embedding_layer(current_tokens)
            _, (decoder_hidden_states, decoder_cell_states) = self.decoder(
                current_embeddings.unsqueeze(1),
                (decoder_hidden_states, decoder_cell_states),
            )
            batch_attention_scores = torch.tanh(
                self.enc_hidden_to_attn(batch_encoder_hidden_states)
                + self.dec_hidden_to_attn(decoder_hidden_states[-1]).unsqueeze(1)
            )
            batch_attention_scores = self.attn_proj(batch_attention_scores).squeeze(2)
            input_padding_mask = batch_input_ids == self.pad_token
            batch_attention_scores = batch_attention_scores.masked_fill(
                input_padding_mask, float("-inf")
            )

            attention_distributions = F.softmax(batch_attention_scores, dim=1)
            context_vectors = torch.bmm(
                attention_distributions.unsqueeze(1), batch_encoder_hidden_states
            ).squeeze(1)
            hidden_contexts = torch.cat(
                [decoder_hidden_states[-1], context_vectors], dim=1
            )

            vocab_distributions = F.softmax(
                self.vocab_proj_2(
                    self.bottle_neck_activation(self.vocab_proj_1(hidden_contexts))
                ),
                dim=1,
            )
            p_gens = torch.sigmoid(
                self.context_to_switch(context_vectors)
                + self.dec_hidden_to_switch(decoder_hidden_states[-1])
                + self.embedding_to_switch(current_embeddings)
            )

            batch_generator_probs = F.pad(
                p_gens * vocab_distributions, (self.end_token, num_oovs)
            )
            batch_pointer_probs = (1 - p_gens) * attention_distributions
            batch_final_distributions = batch_generator_probs.scatter_add(
                1, batch_input_ids, batch_pointer_probs
            )
            batch_final_distributions = F.softmax(batch_final_distributions, dim=1)

            chosen_tokens = beam_search.init_from_first_topk(batch_final_distributions)
            current_embeddings = self._safe_embed(chosen_tokens)
            decoder_hidden_states = decoder_hidden_states.repeat_interleave(
                beam_width, dim=1
            )
            decoder_cell_states = decoder_cell_states.repeat_interleave(
                beam_width, dim=1
            )
            coverage_vectors = attention_distributions.repeat_interleave(
                beam_width, dim=0
            )
            if return_attention:
                attention_distributions_list = [coverage_vectors]
            batch_encoder_hidden_states = batch_encoder_hidden_states.repeat_interleave(
                beam_width, dim=0
            )
            input_padding_mask = input_padding_mask.repeat_interleave(beam_width, dim=0)
            batch_input_ids = batch_input_ids.repeat_interleave(beam_width, dim=0)

            for _ in range(2, max_output_length + 1):
                _, (decoder_hidden_states, decoder_cell_states) = self.decoder(
                    current_embeddings.unsqueeze(1),
                    (
                        decoder_hidden_states,
                        decoder_cell_states,
                    ),
                )

                batch_attention_scores = torch.tanh(
                    self.enc_hidden_to_attn(batch_encoder_hidden_states)
                    + self.dec_hidden_to_attn(decoder_hidden_states[-1]).unsqueeze(1)
                    + self.coverage_to_attn(coverage_vectors.unsqueeze(2))
                )
                batch_attention_scores = self.attn_proj(batch_attention_scores).squeeze(
                    2
                )
                batch_attention_scores = batch_attention_scores.masked_fill(
                    input_padding_mask, float("-inf")
                )

                attention_distributions = F.softmax(batch_attention_scores, dim=1)
                coverage_vectors = coverage_vectors + attention_distributions
                context_vectors = torch.bmm(
                    attention_distributions.unsqueeze(1), batch_encoder_hidden_states
                ).squeeze(1)
                hidden_contexts = torch.cat(
                    [decoder_hidden_states[-1], context_vectors], dim=1
                )
                vocab_distributions = F.softmax(
                    self.vocab_proj_2(
                        self.bottle_neck_activation(self.vocab_proj_1(hidden_contexts))
                    ),
                    dim=1,
                )
                p_gens = torch.sigmoid(
                    self.context_to_switch(context_vectors)
                    + self.dec_hidden_to_switch(decoder_hidden_states[-1])
                    + self.embedding_to_switch(current_embeddings)
                )

                batch_generator_probs = F.pad(
                    p_gens * vocab_distributions, (self.end_token, num_oovs)
                )
                batch_pointer_probs = (1 - p_gens) * attention_distributions
                batch_final_distributions = batch_generator_probs.scatter_add(
                    1, batch_input_ids, batch_pointer_probs
                )

                chosen_tokens, chosen_beam_indices = beam_search.advance(
                    batch_final_distributions,
                    trigram_penalty,
                    bigram_penalty,
                    unigram_penalty,
                    penalty_range,
                )
                current_embeddings = self._safe_embed(chosen_tokens)
                batch_input_ids = batch_input_ids[chosen_beam_indices]

                if return_attention:
                    attention_distributions_list.append(attention_distributions)
                    for i in range(len(attention_distributions_list)):
                        attention_distributions_list[i] = attention_distributions_list[
                            i
                        ][chosen_beam_indices]

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
                        for attention_distributions in attention_distributions_list
                    ],
                    dim=1,
                )

            return output
