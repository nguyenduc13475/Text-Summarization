import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.beam_search import BeamSearch
from src.models.base_model import BaseSummarizationModel
from src.utils.nn_utils import init_weights
from src.utils.utils import create_appearance_boost


class PositionalEncoding(nn.Module):
    """Implementation of Positional Encoding using buffer."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        return x + 0.1 * self.pe[start_pos : start_pos + seq_len].transpose(0, 1)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        return self.pos_encoder(self.embedding(input_ids), start_pos=start_pos)


class EncoderLayerWithAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src, attn_weights


class DecoderLayerWithAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        tgt2, self_attn_weights = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, cross_attn_weights = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        ff = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff)
        tgt = self.norm3(tgt)

        return tgt, self_attn_weights, cross_attn_weights


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayerWithAttn(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayerWithAttn(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.last_encoder_attn = None
        self.last_decoder_cross_attn = None
        self.last_decoder_self_attn = None

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = src
        enc_attn = None
        for layer in self.encoder_layers:
            memory, enc_attn = layer(memory, src_mask, src_key_padding_mask)
        self.last_encoder_attn = enc_attn

        out = tgt
        dec_cross_attn = None
        dec_self_attn = None
        for layer in self.decoder_layers:
            out, self_attn, cross_attn = layer(
                out,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            dec_cross_attn = cross_attn
            dec_self_attn = self_attn

        self.last_decoder_cross_attn = dec_cross_attn
        self.last_decoder_self_attn = dec_self_attn

        return out


class Transformer(BaseSummarizationModel):
    def __init__(
        self,
        tokenizer: Any,
        d_model: int = 256,
        nhead: int = 2,
        num_layers: int = 3,
        learning_rate: float = 1e-4,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.vocab_size = tokenizer.get_vocab_size()
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")
        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.d_model = d_model

        self.embedding_layer = EmbeddingLayer(self.vocab_size, d_model)
        self.transformer = SimpleTransformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=0
        )
        self.hidden_proj_1 = nn.Linear(d_model, d_model * 2)
        self.activation_1 = nn.ReLU()
        self.hidden_proj_2 = nn.Linear(d_model * 2, d_model * 2)
        self.activation_2 = nn.ReLU()
        self.out_proj = nn.Linear(d_model * 2, self.vocab_size - self.end_token)
        self.device = torch.device(device)

        self.apply(init_weights)
        self.to(device)
        self.setup_training_env(learning_rate=learning_rate, loss_scale=1e-3)

    def final_project(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(
            self.activation_2(
                self.hidden_proj_2(self.activation_1(self.hidden_proj_1(x)))
            )
        )

    def compute_loss(
        self,
        batch_input_ids: torch.Tensor,
        batch_target_ids: torch.Tensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        batch_input_ids = batch_input_ids.to(self.device)
        batch_target_ids = batch_target_ids.to(self.device)

        batch_size = batch_input_ids.shape[0]
        max_target_length = batch_target_ids.shape[1]
        batch_current_tokens = torch.cat(
            [
                torch.full((batch_size, 1), self.start_token, device=self.device),
                batch_target_ids[:, :-1],
            ],
            dim=1,
        )
        batch_input_embeddings = self.embedding_layer(batch_input_ids)
        batch_current_token_embeddings = self.embedding_layer(batch_current_tokens)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            max_target_length, dtype=torch.bool
        ).to(self.device)
        input_padding_mask = batch_input_ids == self.pad_token
        current_token_padding_mask = batch_current_tokens == self.pad_token

        decoder_outputs = self.transformer(
            batch_input_embeddings,
            batch_current_token_embeddings,
            tgt_mask=causal_mask,
            src_key_padding_mask=input_padding_mask,
            tgt_key_padding_mask=current_token_padding_mask,
            memory_key_padding_mask=input_padding_mask,
        )

        batch_vocab_distributions = F.pad(
            F.softmax(
                self.final_project(decoder_outputs),
                dim=-1,
            ),
            (self.end_token, 0),
        )
        batch_log_probs = torch.log(batch_vocab_distributions + 1e-9).view(
            batch_size * max_target_length, -1
        )
        token_log_probs = batch_log_probs[
            torch.arange(batch_size * max_target_length), batch_target_ids.view(-1)
        ][batch_target_ids.view(-1) != self.pad_token]
        loss = -token_log_probs.sum()

        return {"total_loss": loss}

    def _calculate_ngram_boost(
        self,
        beam_sequences: torch.Tensor,
        batch_input_ids: torch.Tensor,
        original_attention: float,
        max_input_length: int,
        batch_size: int,
        beam_width: int,
    ) -> torch.Tensor:
        """
        Calculate n-gram penalties/boosts to avoid repetition and encourage models
        to stick to the original text structure.
        """
        ngram_boost = torch.zeros(
            batch_size * beam_width,
            self.vocab_size - self.end_token,
            device=self.device,
        )

        for b in range(batch_size * beam_width):
            seq = beam_sequences[b].tolist()
            # Determine the input sequence corresponding to the current beam branch.
            input_tokens = batch_input_ids[b // beam_width].tolist()

            is_continue = True
            for lookpast in range(5, 2, -1):
                if len(seq) >= lookpast and is_continue:
                    recent_tokens = seq[-lookpast:]
                    for j in range(max_input_length - lookpast):
                        # If a matching n-gram is found in the source text.
                        if input_tokens[j : j + len(recent_tokens)] == recent_tokens:
                            next_token = input_tokens[j + len(recent_tokens)]
                            if next_token >= self.end_token:
                                # Boost the probability for the next token.
                                ngram_boost[b, next_token - self.end_token] += (
                                    original_attention * lookpast * 0.5
                                )
                                is_continue = False
        return ngram_boost

    def infer(
        self,
        batch_input_ids: torch.Tensor,
        max_output_length: int = 100,
        beam_width: int = 6,
        trigram_penalty: float = -30.0,
        bigram_penalty: float = -15.0,
        unigram_penalty: float = -2.0,
        penalty_range: int = 15,
        original_attention: float = 2.0,
        return_attention: bool = False,
        return_embedding: bool = False,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            # 1. Setup Inputs & Beam Search
            if not isinstance(batch_input_ids, torch.Tensor):
                batch_input_ids = torch.tensor(
                    batch_input_ids, device=self.device, dtype=torch.long
                )
            batch_input_ids = batch_input_ids.to(self.device)
            beam_width = min(beam_width, self.vocab_size)

            if batch_input_ids.dim() == 1:
                batch_input_ids = batch_input_ids.unsqueeze(0)

            batch_size, max_input_length = batch_input_ids.shape
            beam_search = BeamSearch(
                batch_size, beam_width, self.start_token, self.end_token, self.device
            )

            # 2. Encode Source Text
            batch_input_embeddings = self.embedding_layer(batch_input_ids)
            input_padding_mask = batch_input_ids == self.pad_token

            # Appearance boost: Encourage the model to use words found in the source.
            appearance_boost = create_appearance_boost(
                batch_input_ids, self, original_attention
            )

            # Initialize decoder inputs
            start_token = torch.tensor([self.start_token], device=self.device)
            start_embedding = self.embedding_layer(start_token).unsqueeze(1)
            batch_current_output_embeddings = start_embedding.repeat(batch_size, 1, 1)

            # 3. Unified Autoregressive Loop
            for t in range(max_output_length):
                # Masking for autoregressive generation
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    batch_current_output_embeddings.shape[1], dtype=torch.bool
                ).to(self.device)

                # Expand dimensions for Beam Search if t > 0
                current_input_embeddings = (
                    batch_input_embeddings
                    if t == 0
                    else batch_input_embeddings.repeat_interleave(beam_width, dim=0)
                )
                current_padding_mask = (
                    input_padding_mask
                    if t == 0
                    else input_padding_mask.repeat_interleave(beam_width, dim=0)
                )
                current_app_boost = (
                    appearance_boost
                    if t == 0
                    else appearance_boost.repeat_interleave(beam_width, dim=0)
                )

                # Decoder forward pass
                decoder_outputs = self.transformer(
                    current_input_embeddings,
                    batch_current_output_embeddings,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=current_padding_mask,
                    memory_key_padding_mask=current_padding_mask,
                )[
                    :, -1, :
                ]  # Only retrieve the output of the last token

                # N-gram boost logic
                ngram_boost = (
                    0
                    if t == 0
                    else self._calculate_ngram_boost(
                        beam_sequences=beam_search.sequences,
                        batch_input_ids=batch_input_ids,
                        original_attention=original_attention,
                        max_input_length=max_input_length,
                        batch_size=batch_size,
                        beam_width=beam_width,
                    )
                )

                # Calculate Vocabulary Distribution
                vocab_distributions = F.pad(
                    F.softmax(
                        self.final_project(decoder_outputs)
                        + current_app_boost
                        + ngram_boost,
                        dim=-1,
                    ),
                    (self.end_token, 0),
                )

                # Beam Search Advancement
                if t == 0:
                    chosen_tokens = beam_search.init_from_first_topk(
                        vocab_distributions
                    )

                    # Update context embeddings for the next step
                    batch_current_output_embeddings = torch.cat(
                        [
                            start_embedding.repeat(batch_size * beam_width, 1, 1),
                            self.embedding_layer(chosen_tokens, start_pos=1).unsqueeze(
                                1
                            ),
                        ],
                        dim=1,
                    )
                else:
                    chosen_tokens, chosen_beam_indices = beam_search.advance(
                        vocab_distributions,
                        trigram_penalty,
                        bigram_penalty,
                        unigram_penalty,
                        penalty_range,
                    )

                    current_length = batch_current_output_embeddings.shape[1]
                    batch_current_output_embeddings = torch.cat(
                        [
                            batch_current_output_embeddings[chosen_beam_indices],
                            self.embedding_layer(
                                chosen_tokens, start_pos=current_length
                            ).unsqueeze(1),
                        ],
                        dim=1,
                    )

                if beam_search.finishes.all():
                    break

            # 4. Finalize Outputs
            chosen_beam_indices = beam_search.finalize_best_beams()
            output = {"output_ids": beam_search.sequences[chosen_beam_indices, 1:]}

            if return_embedding:
                output["input_embeddings"] = batch_input_embeddings
            if return_attention:
                output["encoder_self_attention_distributions"] = (
                    self.transformer.last_encoder_attn[chosen_beam_indices]
                )
                output["decoder_self_attention_distributions"] = (
                    self.transformer.last_decoder_self_attn[chosen_beam_indices]
                )
                output["cross_attention_distributions"] = (
                    self.transformer.last_decoder_cross_attn[chosen_beam_indices]
                )

            return output
