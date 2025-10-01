import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from beam_search import BeamSearch
from utils import tensor_dict_to_scalar


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    @classmethod
    def positional_encoding(cls, max_sequence_length, d_model):
        positions = torch.arange(max_sequence_length).unsqueeze(1)
        dims = torch.arange(d_model).unsqueeze(0)

        angle_rates = 1 / torch.pow(10000, (2 * (dims // 2)) / d_model)
        angle_rads = positions * angle_rates

        pos_encoding = torch.zeros_like(angle_rads)
        pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return pos_encoding

    def forward(self, input_ids):
        return self.embedding(input_ids) + self.positional_encoding(
            input_ids.shape[-1], self.embedding_dim
        ).to(input_ids.device)


class EncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
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
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
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
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
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
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
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
        src,
        tgt,
        tgt_mask=None,
        src_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
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


class Transformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        d_model=512,
        nhead=8,
        num_layers=6,
        learning_rate=1e-3,
        device="cpu",
    ):
        super().__init__()
        self.vocab_size = tokenizer.get_vocab_size()
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")
        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.d_model = d_model

        self.embedding_layer = EmbeddingLayer(self.vocab_size, d_model)
        self.transformer = SimpleTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.out_proj = nn.Linear(d_model, self.vocab_size - self.end_token)
        self.device = torch.device(device)

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_scale = 1e-2

    def _safe_ids(self, ids):
        return torch.where(
            ids >= self.vocab_size,
            torch.tensor(self.unknown_token, device=self.device),
            ids.to,
        )

    def _safe_embed(self, ids):
        return self.embedding_layer(self._safe_ids(ids))

    def make_padding_mask(self, batch_input_ids):
        mask = batch_input_ids.eq(self.pad_token)
        return mask.float().masked_fill(mask, float("-inf"))

    def compute_loss(self, batch_input_ids, batch_target_ids):
        batch_input_ids = batch_input_ids.to(self.device)
        batch_target_ids = batch_target_ids.to(self.device)

        batch_size = batch_input_ids.shape[0]
        max_target_length = batch_target_ids.shape[1]
        batch_input_ids = self._safe_ids(batch_input_ids)
        batch_target_ids = self._safe_ids(batch_target_ids)
        batch_current_tokens = torch.cat(
            [
                torch.full((batch_size, 1), self.start_token, device=self.device),
                batch_target_ids,
            ],
            dim=1,
        )
        batch_next_tokens = torch.cat(
            [
                batch_target_ids,
                torch.full((batch_size, 1), self.pad_token, device=self.device),
            ],
            dim=1,
        )
        batch_next_tokens[
            torch.arange(batch_size, device=self.device),
            (batch_next_tokens == self.pad_token).int().argmax(dim=1),
        ] = self.end_token

        batch_input_embeddings = self.embedding_layer(batch_input_ids)
        batch_current_token_embeddings = self.embedding_layer(batch_current_tokens)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            max_target_length + 1
        ).to(self.device)
        input_padding_mask = self.make_padding_mask(batch_input_ids)
        current_token_padding_mask = self.make_padding_mask(batch_current_tokens)

        decoder_outputs = self.transformer(
            batch_input_embeddings,
            batch_current_token_embeddings,
            tgt_mask=causal_mask,
            src_key_padding_mask=input_padding_mask,
            tgt_key_padding_mask=current_token_padding_mask,
            memory_key_padding_mask=input_padding_mask,
        )

        batch_vocab_distributions = F.pad(
            F.softmax(self.out_proj(decoder_outputs), dim=2), (self.end_token, 0)
        )
        batch_log_probs = torch.log(batch_vocab_distributions + 1e-9)

        smoothing = 0.1
        batch_target_distributions = torch.zeros_like(batch_vocab_distributions)
        batch_target_distributions.fill_(smoothing / (self.vocab_size - 1))
        batch_target_distributions.scatter_(
            2, batch_next_tokens.unsqueeze(2), 1.0 - smoothing
        )

        batch_target_distributions.masked_fill_(
            batch_next_tokens.unsqueeze(2) == self.pad_token, 0.0
        )

        loss = F.kl_div(
            batch_log_probs,
            batch_target_distributions,
            reduction="sum",
        )

        return {"total_loss": loss}

    def train_one_batch(self, batch_input_ids, batch_target_ids):
        self.train()
        losses = self.compute_loss(batch_input_ids, batch_target_ids)

        self.optimizer.zero_grad()
        (losses["total_loss"] * self.loss_scale).backward()
        self.optimizer.step()

        return tensor_dict_to_scalar(losses)

    def validate_one_batch(self, batch_input_ids, batch_target_ids):
        self.eval()
        with torch.no_grad():
            losses = self.compute_loss(batch_input_ids, batch_target_ids)
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
            batch_input_ids = batch_input_ids.to(self.device)
            beam_width = min(beam_width, self.vocab_size)

            if batch_input_ids.dim() == 1:
                batch_input_ids = batch_input_ids.unsqueeze(0)

            batch_size = batch_input_ids.shape[0]
            beam_search = BeamSearch(
                batch_size, beam_width, self.start_token, self.end_token, self.device
            )

            batch_input_embeddings = self._safe_embed(batch_input_ids)

            input_padding_mask = self.make_padding_mask(batch_input_ids)

            start_token = torch.tensor([self.start_token], device=self.device)
            start_embedding = self.embedding_layer(start_token).unsqueeze(1)

            batch_current_output_embeddings = start_embedding.repeat(batch_size, 1, 1)

            decoder_outputs = self.transformer(
                batch_input_embeddings,
                batch_current_output_embeddings,
                src_key_padding_mask=input_padding_mask,
                memory_key_padding_mask=input_padding_mask,
            )[:, -1, :]

            vocab_distributions = F.pad(
                F.softmax(self.out_proj(decoder_outputs), dim=1),
                (self.end_token, 0),
            )

            chosen_tokens = beam_search.init_from_first_topk(vocab_distributions)

            batch_current_output_embeddings = torch.cat(
                [
                    start_embedding.repeat(batch_size * beam_width, 1, 1),
                    self.embedding_layer(chosen_tokens).unsqueeze(1),
                ],
                dim=1,
            )

            beam_input_embeddings = batch_input_embeddings.repeat_interleave(
                beam_width, dim=0
            )
            input_padding_mask = input_padding_mask.repeat_interleave(beam_width, dim=0)

            for _ in range(2, max_output_length + 1):
                decoder_outputs = self.transformer(
                    beam_input_embeddings,
                    batch_current_output_embeddings,
                    src_key_padding_mask=input_padding_mask,
                    memory_key_padding_mask=input_padding_mask,
                )[:, -1, :]

                vocab_distributions = F.pad(
                    F.softmax(self.out_proj(decoder_outputs), dim=1),
                    (self.end_token, 0),
                )
                chosen_tokens, chosen_beam_indices = beam_search.advance(
                    vocab_distributions
                )

                batch_current_output_embeddings = torch.cat(
                    [
                        batch_current_output_embeddings[chosen_beam_indices],
                        self.embedding_layer(chosen_tokens).unsqueeze(1),
                    ],
                    dim=1,
                )

                if beam_search.finishes.all():
                    break

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
