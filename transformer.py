import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from beam_search import beam_search


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


class TransformerSummarizer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        start_token=1,
        end_token=2,
        unknown_token=3,
        pad_token=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.d_model = d_model

        self.embedding_layer = EmbeddingLayer(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def make_padding_mask(self, input_ids):
        mask = input_ids.eq(self.pad_token)
        return mask.float().masked_fill(mask, float("-inf"))

    def train_one_batch(self, input_ids, labels, input_lengths=None):
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

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def infer(self, input_ids, max_summary_length=100, beam_width=4):
        input_embeddings = self.embedding_layer(
            torch.where(
                input_ids >= self.vocab_size,
                torch.tensor(self.unknown_token),
                input_ids,
            )
        )

        # {current_output_embeddings}
        def predictor(state):
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

            return vocab_distribution, new_state

        summary = beam_search(
            predictor=predictor,
            start_state={
                "current_output_embeddings": self.embedding_layer(
                    torch.tensor([self.start_token])
                ),
                "sequence": [self.start_token],
            },
            beam_width=beam_width,
            max_state_length=max_summary_length,
            end_state_indicator=lambda state: state["sequence"][-1] == self.end_token,
        )["sequence"]

        return summary[1:]
