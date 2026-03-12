from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.core.beam_search import BeamSearch
from src.models.base_model import BaseSummarizationModel
from src.utils.nn_utils import init_weights


class PointerGeneratorNetwork(BaseSummarizationModel):
    def __init__(
        self,
        tokenizer: Any,
        embedding_dim: int = 128,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        attention_dim: int = 256,
        bottle_neck_dim: int = 512,
        num_layers: int = 2,
        cov_loss_factor: float = 1.0,
        learning_rate: float = 1e-3,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(device)
        self.vocab_size = tokenizer.get_vocab_size()
        self.num_layers = num_layers

        # Core components
        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim,
            encoder_hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )
        self.decoder = nn.LSTM(
            embedding_dim, decoder_hidden_dim, batch_first=True, num_layers=num_layers
        )

        # State Projections
        self.enc_to_dec_hidden = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.enc_to_dec_cell = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

        # Attention & Pointer Mechanism
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
        self.vocab_proj_2 = nn.Linear(
            bottle_neck_dim, self.vocab_size - tokenizer.token_to_id("</s>")
        )

        self.context_to_switch = nn.Linear(encoder_hidden_dim * 2, 1)
        self.dec_hidden_to_switch = nn.Linear(decoder_hidden_dim, 1, bias=False)
        self.embedding_to_switch = nn.Linear(embedding_dim, 1, bias=False)

        # Tokens
        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")

        self.apply(init_weights)
        self.to(device)
        self.cov_loss_factor = cov_loss_factor
        self.setup_training_env(learning_rate=learning_rate, loss_scale=1e-2)

    def _encode(
        self, batch_input_ids: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared encoder logic for both training and inference."""
        batch_embeddings = self._safe_embed(batch_input_ids)
        packed_seq = pack_padded_sequence(
            batch_embeddings,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoder_states, (final_hidden, final_cell) = self.encoder(packed_seq)
        encoder_states, _ = pad_packed_sequence(encoder_states, batch_first=True)

        batch_size = batch_input_ids.shape[0]
        # Reshape bidirectional outputs safely
        final_hidden = (
            final_hidden.view(self.num_layers, 2, batch_size, -1)
            .transpose(1, 2)
            .reshape(self.num_layers, batch_size, -1)
        )
        final_cell = (
            final_cell.view(self.num_layers, 2, batch_size, -1)
            .transpose(1, 2)
            .reshape(self.num_layers, batch_size, -1)
        )

        decoder_hidden = self.enc_to_dec_hidden(final_hidden)
        decoder_cell = self.enc_to_dec_cell(final_cell)

        return encoder_states, decoder_hidden, decoder_cell

    def _decoder_step(
        self,
        current_embeddings: torch.Tensor,
        decoder_hidden: torch.Tensor,
        decoder_cell: torch.Tensor,
        encoder_states: torch.Tensor,
        coverage_vectors: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes a single step of the decoder, returning distributions and states."""
        _, (decoder_hidden, decoder_cell) = self.decoder(
            current_embeddings.unsqueeze(1), (decoder_hidden, decoder_cell)
        )

        # Calculate Attention
        attn_scores = torch.tanh(
            self.enc_hidden_to_attn(encoder_states)
            + self.dec_hidden_to_attn(decoder_hidden[-1]).unsqueeze(1)
            + self.coverage_to_attn(coverage_vectors.unsqueeze(2))
        )
        attn_scores = self.attn_proj(attn_scores).squeeze(2)

        if input_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(input_padding_mask, float("-inf"))

        attn_dists = F.softmax(attn_scores, dim=1)
        context_vectors = torch.bmm(attn_dists.unsqueeze(1), encoder_states).squeeze(1)

        # Vocabulary Distribution
        hidden_contexts = torch.cat([decoder_hidden[-1], context_vectors], dim=1)
        vocab_dists = F.softmax(
            self.vocab_proj_2(
                self.bottle_neck_activation(self.vocab_proj_1(hidden_contexts))
            ),
            dim=1,
        )

        # Generation Probability (p_gen)
        p_gens = torch.sigmoid(
            self.context_to_switch(context_vectors)
            + self.dec_hidden_to_switch(decoder_hidden[-1])
            + self.embedding_to_switch(current_embeddings)
        ).squeeze(1)

        return vocab_dists, attn_dists, p_gens, decoder_hidden, decoder_cell

    def compute_loss(
        self,
        batch_input_ids: torch.Tensor,
        batch_target_ids: torch.Tensor,
        oov_lists: List[List[str]],
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        batch_input_ids, batch_target_ids = batch_input_ids.to(
            self.device
        ), batch_target_ids.to(self.device)
        batch_size, max_input_length = batch_input_ids.shape
        max_target_length = batch_target_ids.shape[1]
        max_num_oovs = max((len(oov) for oov in oov_lists), default=0)

        # Utilize shared encoder
        encoder_states, dec_hidden, dec_cell = self._encode(
            batch_input_ids, input_lengths
        )

        coverage_vectors = torch.zeros(batch_size, max_input_length, device=self.device)
        current_tokens = torch.full(
            (batch_size,), self.start_token, device=self.device, dtype=torch.long
        )

        nll_losses, cov_losses = torch.zeros(
            batch_size, device=self.device
        ), torch.zeros(batch_size, device=self.device)
        input_padding_mask = batch_input_ids == self.pad_token

        for t in range(max_target_length):
            vocab_dists, attn_dists, p_gens, dec_hidden, dec_cell = self._decoder_step(
                self.embedding_layer(current_tokens),
                dec_hidden,
                dec_cell,
                encoder_states,
                coverage_vectors,
                input_padding_mask,
            )

            next_tokens = batch_target_ids[:, t]
            current_tokens = self._safe_ids(next_tokens)

            # Pointer Mechanism
            next_token_probs = p_gens * F.pad(
                vocab_dists, (self.end_token, max_num_oovs)
            ).gather(1, next_tokens.unsqueeze(1)).squeeze(1) + (1 - p_gens) * (
                attn_dists * (batch_input_ids == next_tokens.unsqueeze(1)).float()
            ).sum(
                dim=1
            )

            # Masked Loss Calculation
            valid_step_mask = (t < target_lengths.to(self.device)).float()
            nll_losses -= torch.log(next_token_probs + 1e-9) * valid_step_mask
            cov_losses += (
                torch.min(attn_dists, coverage_vectors).sum(dim=1) * valid_step_mask
            )

            coverage_vectors = coverage_vectors + attn_dists

        nll_loss, cov_loss = nll_losses.sum(), cov_losses.sum()
        return {
            "nll_loss": nll_loss,
            "cov_loss": cov_loss,
            "total_loss": nll_loss + cov_loss * self.cov_loss_factor,
        }

    def infer(
        self,
        batch_input_ids: torch.Tensor,
        max_output_length: int = 100,
        beam_width: int = 6,
        return_attention: bool = False,
        return_embedding: bool = False,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            batch_input_ids = torch.as_tensor(
                batch_input_ids, device=self.device, dtype=torch.long
            )
            if batch_input_ids.dim() == 1:
                batch_input_ids = batch_input_ids.unsqueeze(0)

            batch_size = batch_input_ids.shape[0]
            beam_width = min(beam_width, self.vocab_size)
            beam_search = BeamSearch(
                batch_size, beam_width, self.start_token, self.end_token, self.device
            )
            num_oovs = max(batch_input_ids.max().item() - self.vocab_size + 1, 0)

            # Utilize shared encoder
            encoder_states, dec_hidden, dec_cell = self._encode(
                batch_input_ids, (batch_input_ids != self.pad_token).sum(dim=1)
            )

            coverage_vectors = torch.zeros(
                batch_size, batch_input_ids.shape[1], device=self.device
            )
            current_tokens = torch.full(
                (batch_size,), self.start_token, device=self.device, dtype=torch.long
            )
            input_padding_mask = batch_input_ids == self.pad_token

            attn_dists_list = []

            for t in range(max_output_length):
                vocab_dists, attn_dists, p_gens, dec_hidden, dec_cell = (
                    self._decoder_step(
                        self.embedding_layer(current_tokens),
                        dec_hidden,
                        dec_cell,
                        encoder_states,
                        coverage_vectors,
                        input_padding_mask,
                    )
                )

                gen_probs = F.pad(
                    p_gens.unsqueeze(1) * vocab_dists, (self.end_token, num_oovs)
                )
                ptr_probs = (1 - p_gens).unsqueeze(1) * attn_dists
                final_dists = F.softmax(
                    gen_probs.scatter_add(1, batch_input_ids, ptr_probs), dim=1
                )

                if t == 0:
                    chosen_tokens = beam_search.init_from_first_topk(final_dists)
                    # Expand states for beam search
                    dec_hidden = dec_hidden.repeat_interleave(beam_width, dim=1)
                    dec_cell = dec_cell.repeat_interleave(beam_width, dim=1)
                    encoder_states = encoder_states.repeat_interleave(beam_width, dim=0)
                    coverage_vectors = attn_dists.repeat_interleave(beam_width, dim=0)
                    input_padding_mask = input_padding_mask.repeat_interleave(
                        beam_width, dim=0
                    )
                    batch_input_ids = batch_input_ids.repeat_interleave(
                        beam_width, dim=0
                    )
                    if return_attention:
                        attn_dists_list = [coverage_vectors]
                else:
                    coverage_vectors += attn_dists
                    chosen_tokens, chosen_beam_idx = beam_search.advance(
                        final_dists, **kwargs
                    )
                    dec_hidden = dec_hidden[:, chosen_beam_idx, :]
                    dec_cell = dec_cell[:, chosen_beam_idx, :]
                    coverage_vectors = coverage_vectors[chosen_beam_idx]
                    batch_input_ids = batch_input_ids[chosen_beam_idx]
                    if return_attention:
                        attn_dists_list.append(attn_dists)
                        attn_dists_list = [d[chosen_beam_idx] for d in attn_dists_list]

                current_tokens = self._safe_embed(
                    chosen_tokens
                )  # Need token index for next embedding
                current_tokens = chosen_tokens  # Using IDs safely

                if beam_search.finishes.all():
                    break

            best_beam_idx = beam_search.finalize_best_beams()
            output = {"output_ids": beam_search.sequences[best_beam_idx, 1:]}

            if return_embedding:
                output["input_embeddings"] = self._safe_embed(
                    batch_input_ids[best_beam_idx]
                )
            if return_attention:
                output["cross_attention_distributions"] = torch.stack(
                    [d[best_beam_idx] for d in attn_dists_list], dim=1
                )

            return output
