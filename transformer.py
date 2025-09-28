import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from beam_search import beam_search
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
        )


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
        # self-attention
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
        # feed forward
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src, attn_weights


class DecoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )  # cross-attn
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
        # tgt self-attention
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

        # cross-attention: queries = tgt, keys/values = memory
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

        # feed forward
        ff = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff)
        tgt = self.norm3(tgt)

        # return output, self-attn, cross-attn
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
        # to be set after forward
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
        # src: (batch, src_len, d_model)  (we assume batch_first=True)
        # tgt: (batch, tgt_len, d_model)
        memory = src
        enc_attn = None
        for layer in self.encoder_layers:
            memory, enc_attn = layer(memory, src_mask, src_key_padding_mask)
        # memory is (batch, src_len, d_model)
        self.last_encoder_attn = enc_attn  # (batch, heads, src_len, src_len) or None

        out = tgt
        dec_cross_attn = None
        dec_self_attn = None  # <-- thêm: container cho self-attn của layer cuối
        for layer in self.decoder_layers:
            out, self_attn, cross_attn = layer(
                out,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            dec_cross_attn = cross_attn  # last layer's cross-attn
            dec_self_attn = self_attn  # <-- thêm: keep last layer's self-attn

        self.last_decoder_cross_attn = dec_cross_attn
        self.last_decoder_self_attn = (
            dec_self_attn  # <-- thêm: expose decoder self-attn
        )

        # return decoder outputs (batch, tgt_len, d_model)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        tokenizer,
        d_model=512,
        nhead=8,
        num_layers=6,
        learning_rate=1e-3,
        device="cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")
        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.d_model = d_model

        self.embedding_layer = EmbeddingLayer(vocab_size, d_model)
        self.transformer = SimpleTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def make_padding_mask(self, input_ids):
        mask = input_ids.eq(self.pad_token)
        return mask.float().masked_fill(mask, float("-inf"))

    def compute_loss(self, input_ids, labels, input_lengths=None):
        batch_size = len(input_ids)
        input_ids = torch.where(
            input_ids >= self.vocab_size,
            torch.tensor(self.unknown_token),
            input_ids,
        )
        labels = torch.where(
            labels >= self.vocab_size,
            torch.tensor(self.unknown_token),
            labels,
        )
        current_output_ids = torch.cat(
            [torch.full((batch_size, 1), self.start_token), labels], dim=1
        )
        next_output_ids = torch.cat(
            [labels, torch.full((batch_size, 1), self.pad_token)], dim=1
        )
        next_output_ids[
            torch.arange(batch_size),
            (next_output_ids == self.pad_token).int().argmax(dim=1),
        ] = self.end_token

        input_embeddings = self.embedding_layer(input_ids)
        current_output_embeddings = self.embedding_layer(current_output_ids)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            current_output_ids.shape[-1]
        )
        input_padding_mask = self.make_padding_mask(input_ids)
        current_output_padding_mask = self.make_padding_mask(current_output_ids)

        next_output_embeddings = self.transformer(
            input_embeddings,
            current_output_embeddings,
            tgt_mask=causal_mask,
            src_key_padding_mask=input_padding_mask,
            tgt_key_padding_mask=current_output_padding_mask,
            memory_key_padding_mask=input_padding_mask,
        )

        vocab_distribution = F.softmax(self.out_proj(next_output_embeddings), dim=2)
        log_probs = torch.log(vocab_distribution + 1e-9)

        smoothing = 0.1
        smooth_distribution = torch.zeros_like(vocab_distribution)
        smooth_distribution.fill_(smoothing / (self.vocab_size - 1))
        smooth_distribution.scatter_(2, next_output_ids.unsqueeze(2), 1.0 - smoothing)

        smooth_distribution.masked_fill_(
            next_output_ids.unsqueeze(2) == self.pad_token, 0.0
        )

        loss = (
            F.kl_div(
                log_probs,
                smooth_distribution,
                reduction="batchmean",
            )
            * 0.01
        )

        return {"total_loss": loss}

    def train_one_batch(self, input_ids, labels, input_lengths=None):
        losses = self.compute_loss(input_ids, labels, input_lengths)

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()

        return tensor_dict_to_scalar(losses)

    def validate_one_batch(self, input_ids, labels, input_lengths=None):
        with torch.no_grad():
            losses = self.compute_loss(input_ids, labels, input_lengths)
            return tensor_dict_to_scalar(losses)

    def infer(
        self,
        input_ids,
        max_summary_length=100,
        beam_width=4,
        return_attention=False,
        return_embedding=False,
    ):
        input_embeddings = self.embedding_layer(
            torch.where(
                input_ids >= self.vocab_size,
                torch.tensor(self.unknown_token),
                input_ids,
            )
        )

        # helper to pick head-0 / batch-0 slice robustly
        def _pick_head0(att_tensor):
            if att_tensor is None:
                return None
            at = att_tensor
            # possible shapes: (batch, heads, t, s) or (heads, t, s) or (batch, t, s)
            # prefer to return (t, s) for head 0.
            try:
                # if (batch, heads, t, s) -> take batch 0, head 0 -> (t, s)
                return at[0, 0].detach().cpu().clone()
            except Exception:
                try:
                    # if (heads, t, s) -> take head 0
                    return at[0].detach().cpu().clone()
                except Exception:
                    try:
                        # if (batch, t, s) -> take batch 0
                        return at[0].detach().cpu().clone()
                    except Exception:
                        return at.detach().cpu().clone()

        # {current_output_embeddings}
        def predictor(state):
            nonlocal return_attention
            new_state = dict()
            next_token_embedding = self.transformer(
                input_embeddings, state["current_output_embeddings"]
            )[-1]
            new_state["current_output_embeddings"] = torch.cat(
                [state["current_output_embeddings"], next_token_embedding.unsqueeze(0)],
                dim=0,
            )
            vocab_distribution = F.softmax(
                self.out_proj(next_token_embedding),
                dim=0,
            )

            if return_attention:
                enc_attn = _pick_head0(self.transformer.last_encoder_attn)
                dec_attn = _pick_head0(self.transformer.last_decoder_self_attn)
                cross_attn = _pick_head0(self.transformer.last_decoder_cross_attn)

                new_state["input_attention_distributions"] = state[
                    "input_attention_distributions"
                ] + [enc_attn]
                new_state["output_attention_distributions"] = state[
                    "output_attention_distributions"
                ] + [dec_attn]
                new_state["cross_attention_distributions"] = state[
                    "cross_attention_distributions"
                ] + [cross_attn]

            return vocab_distribution, new_state

        beam_search_final_state = beam_search(
            predictor=predictor,
            start_state={
                "current_output_embeddings": self.embedding_layer(
                    torch.tensor([self.start_token])
                ),
                "sequence": [self.start_token],
            }
            | (
                {
                    "input_attention_distributions": [],
                    "output_attention_distributions": [],
                    "cross_attention_distributions": [],
                }
                if return_attention
                else {}
            ),
            beam_width=beam_width,
            max_state_length=max_summary_length,
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
                    "output_attention_distributions": (
                        beam_search_final_state["output_attention_distributions"]
                    ),
                    "cross_attention_distributions": (
                        beam_search_final_state["cross_attention_distributions"]
                    ),
                }
                if return_attention
                else {}
            )
            | ({"embedding": input_embeddings} if return_embedding else None)
        )
