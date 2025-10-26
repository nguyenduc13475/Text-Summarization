import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from beam_search import BeamSearch
from utils import tensor_dict_to_scalar


def init_weights(m):
    if isinstance(m, nn.Embedding):
        init.uniform_(m.weight, -0.1, 0.1)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            init.xavier_uniform_(m.in_proj_weight)
        if m.out_proj.weight is not None:
            init.xavier_uniform_(m.out_proj.weight)
        if m.in_proj_bias is not None:
            init.zeros_(m.in_proj_bias)
        if m.out_proj.bias is not None:
            init.zeros_(m.out_proj.bias)

    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        init.zeros_(m.bias)


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
        return self.embedding(input_ids) + 0.1 * self.positional_encoding(
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
        d_model=256,
        nhead=2,
        num_layers=3,
        learning_rate=1e-4,
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
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=0
        )
        self.out_proj = nn.Linear(d_model, self.vocab_size - self.end_token)
        self.device = torch.device(device)

        self.apply(init_weights)
        self.to(device)
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        self.loss_scale = 1e-3

    def compute_loss(self, batch_input_ids, batch_target_ids):
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
                self.out_proj(decoder_outputs),
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

    def train_one_batch(self, batch_input_ids, batch_target_ids):
        self.train()
        self.optimizer.zero_grad()

        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                losses = self.compute_loss(batch_input_ids, batch_target_ids)
            self.scaler.scale(losses["total_loss"] * self.loss_scale).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses = self.compute_loss(batch_input_ids, batch_target_ids)
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
        beam_width=1,
        trigram_penalty=-1e5,
        original_attention=0.7,
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

            batch_input_embeddings = self.embedding_layer(batch_input_ids)

            input_padding_mask = batch_input_ids == self.pad_token

            start_token = torch.tensor([self.start_token], device=self.device)
            start_embedding = self.embedding_layer(start_token).unsqueeze(1)

            batch_current_output_embeddings = start_embedding.repeat(batch_size, 1, 1)

            logits_boost = torch.zeros(batch_size, self.vocab_size, device=self.device)
            for i in range(batch_size):
                logits_boost[i, torch.unique(batch_input_ids[i])] = original_attention
            logits_boost = logits_boost[:, self.end_token :]

            decoder_outputs = self.transformer(
                batch_input_embeddings,
                batch_current_output_embeddings,
                src_key_padding_mask=input_padding_mask,
                memory_key_padding_mask=input_padding_mask,
            )[:, -1, :]

            vocab_distributions = F.pad(
                F.softmax(
                    self.out_proj(decoder_outputs) + logits_boost,
                    dim=-1,
                ),
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
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    batch_current_output_embeddings.shape[1], dtype=torch.bool
                ).to(self.device)

                decoder_outputs = self.transformer(
                    beam_input_embeddings,
                    batch_current_output_embeddings,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=input_padding_mask,
                    memory_key_padding_mask=input_padding_mask,
                )[:, -1, :]

                vocab_distributions = F.pad(
                    F.softmax(
                        self.out_proj(decoder_outputs) + logits_boost,
                        dim=-1,
                    ),
                    (self.end_token, 0),
                )
                chosen_tokens, chosen_beam_indices = beam_search.advance(
                    vocab_distributions, trigram_penalty
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
