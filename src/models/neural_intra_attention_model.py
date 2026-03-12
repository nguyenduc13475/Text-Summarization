from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.core.beam_search import BeamSearch
from src.models.base_model import BaseSummarizationModel
from src.utils.metrics import compute_metric
from src.utils.nn_utils import init_weights
from src.utils.utils import tensor_dict_to_scalar, token_ids_to_text


class NeuralIntraAttentionModel(BaseSummarizationModel):
    def __init__(
        self,
        tokenizer: Any,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        bottle_neck_dim: int = 512,
        num_layers: int = 2,
        rl_loss_factor: float = 1.0,
        learning_rate: float = 1e-3,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(device)
        self.vocab_size, self.num_layers, self.hidden_dim = (
            tokenizer.get_vocab_size(),
            num_layers,
            hidden_dim,
        )
        self.tokenizer = tokenizer

        # Tokens
        self.unknown_token = tokenizer.token_to_id("<unk>")
        self.start_token = tokenizer.token_to_id("<s>")
        self.end_token = tokenizer.token_to_id("</s>")
        self.pad_token = tokenizer.token_to_id("<pad>")

        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )
        self.decoder = nn.LSTM(
            embedding_dim, hidden_dim * 2, batch_first=True, num_layers=num_layers
        )

        self.encoder_attn_proj = nn.Parameter(
            torch.empty(hidden_dim * 2, hidden_dim * 2)
        )
        self.decoder_attn_proj = nn.Parameter(
            torch.empty(hidden_dim * 2, hidden_dim * 2)
        )
        torch.nn.init.xavier_uniform_(self.encoder_attn_proj)
        torch.nn.init.xavier_uniform_(self.decoder_attn_proj)

        self.vocab_proj_1 = nn.Linear(6 * hidden_dim, bottle_neck_dim)
        self.bottle_neck_activation = nn.ReLU()
        self.vocab_proj_2 = nn.Linear(bottle_neck_dim, self.vocab_size - self.end_token)
        self.concat_state_to_switch = nn.Linear(6 * hidden_dim, 1)

        self.apply(init_weights)
        self.to(device)
        self.rl_loss_factor = rl_loss_factor
        self.setup_training_env(learning_rate=learning_rate, loss_scale=1e-3)

    def _encode(
        self, batch_input_ids: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared encoder logic."""
        batch_embeddings = self._safe_embed(batch_input_ids)
        packed_seq = pack_padded_sequence(
            batch_embeddings,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        enc_states, (final_hidden, _) = self.encoder(packed_seq)
        enc_states, _ = pad_packed_sequence(enc_states, batch_first=True)

        batch_size = batch_input_ids.shape[0]
        final_hidden = (
            final_hidden.view(self.num_layers, 2, batch_size, -1)
            .transpose(1, 2)
            .reshape(self.num_layers, batch_size, -1)
        )
        dec_cell = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim * 2, device=self.device
        )

        return enc_states, final_hidden, dec_cell

    # (Giữ nguyên hàm _decoder_step hiện tại của bạn vì nó xử lý toán học Attention rất tốt)
    def _decoder_step(
        self,
        current_embeddings: torch.Tensor,
        decoder_hidden_states: torch.Tensor,
        decoder_cell_states: torch.Tensor,
        batch_encoder_hidden_states: torch.Tensor,
        batch_cummulative_encoder_attention_scores: torch.Tensor,
        previous_decoder_hidden_states: torch.Tensor,
        input_padding_mask: torch.Tensor,
        is_first_step: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _, (decoder_hidden_states, decoder_cell_states) = self.decoder(
            current_embeddings.unsqueeze(1),
            (decoder_hidden_states, decoder_cell_states),
        )

        batch_encoder_attention_scores = (
            (
                (decoder_hidden_states[-1] @ self.encoder_attn_proj).unsqueeze(1)
                @ batch_encoder_hidden_states.transpose(1, 2)
            )
            .squeeze(1)
            .clamp(-10, 10)
        )
        batch_encoder_temporal_scores = (
            torch.exp(batch_encoder_attention_scores)
            if is_first_step
            else torch.exp(batch_encoder_attention_scores)
            / (batch_cummulative_encoder_attention_scores + 1e-5)
        )
        batch_encoder_temporal_scores = batch_encoder_temporal_scores.masked_fill(
            input_padding_mask, 0.0
        )
        encoder_attention_distributions = batch_encoder_temporal_scores / (
            batch_encoder_temporal_scores.sum(dim=1, keepdim=True) + 1e-9
        )
        encoder_context_vectors = torch.bmm(
            encoder_attention_distributions.unsqueeze(1), batch_encoder_hidden_states
        ).squeeze(1)
        batch_cummulative_encoder_attention_scores += torch.exp(
            batch_encoder_attention_scores
        )

        batch_size = current_embeddings.size(0)
        if is_first_step:
            decoder_context_vectors = torch.zeros(
                batch_size, self.hidden_dim * 2, device=self.device
            )
            decoder_attention_distributions = None
        else:
            batch_decoder_attention_scores = (
                (decoder_hidden_states[-1] @ self.decoder_attn_proj).unsqueeze(1)
                @ previous_decoder_hidden_states.transpose(1, 2)
            ).squeeze(1)
            decoder_attention_distributions = F.softmax(
                batch_decoder_attention_scores, dim=1
            )
            decoder_context_vectors = torch.bmm(
                decoder_attention_distributions.unsqueeze(1),
                previous_decoder_hidden_states,
            ).squeeze(1)

        previous_decoder_hidden_states = torch.cat(
            [previous_decoder_hidden_states, decoder_hidden_states[-1].unsqueeze(1)],
            dim=1,
        )
        concat_states = torch.cat(
            [
                decoder_hidden_states[-1],
                encoder_context_vectors,
                decoder_context_vectors,
            ],
            dim=1,
        )

        vocab_distributions = F.softmax(
            self.vocab_proj_2(
                self.bottle_neck_activation(self.vocab_proj_1(concat_states))
            ),
            dim=1,
        )
        p_gens = torch.sigmoid(self.concat_state_to_switch(concat_states)).squeeze(1)

        return (
            vocab_distributions,
            encoder_attention_distributions,
            decoder_attention_distributions,
            p_gens,
            decoder_hidden_states,
            decoder_cell_states,
            batch_cummulative_encoder_attention_scores,
            previous_decoder_hidden_states,
        )

    def compute_loss(
        self,
        batch_input_ids: torch.Tensor,
        batch_target_ids: torch.Tensor,
        oov_lists: List[List[str]],
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        target_texts: Optional[List[str]] = None,
        return_rl_loss: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_input_ids = batch_input_ids.to(self.device)
        batch_target_ids = batch_target_ids.to(self.device)
        target_lengths = target_lengths.to(self.device)
        batch_size, max_input_length = batch_input_ids.shape
        max_target_length = batch_target_ids.shape[1]
        if return_rl_loss:
            max_target_length = min(max_target_length, 50)

        max_num_oovs = 0
        for oov_list in oov_lists:
            if len(oov_list) > max_num_oovs:
                max_num_oovs = len(oov_list)

        batch_encoder_hidden_states, initial_hidden, initial_cell = self._encode(
            batch_input_ids, input_lengths
        )

        if return_rl_loss:
            sampling_sequence_metrics = []
            greedy_sequence_metrics = []
            modes = ["sampling", "greedy"]
        else:
            nll_losses = torch.zeros(batch_size, device=self.device)
            modes = ["teacher"]

        for mode in modes:
            # Clone state to prevent modes (greedy, sampling) from overwriting each other.
            decoder_hidden_states = initial_hidden.clone()
            decoder_cell_states = initial_cell.clone()
            current_tokens = torch.tensor(
                [self.start_token] * batch_size, device=self.device
            )
            batch_cummulative_encoder_attention_scores = torch.zeros(
                batch_size, max_input_length, device=self.device
            )
            previous_decoder_hidden_states = torch.empty(
                batch_size, 0, self.hidden_dim * 2, device=self.device
            )

            if return_rl_loss:
                is_continues = torch.ones(batch_size, device=self.device)

            if mode == "sampling":
                cummulative_sampling_log_probs = torch.zeros(
                    batch_size, device=self.device
                )
                sampling_sequences = torch.empty(
                    batch_size, 0, device=self.device, dtype=torch.long
                )
            elif mode == "greedy":
                greedy_sequences = torch.empty(
                    batch_size, 0, device=self.device, dtype=torch.long
                )

            input_padding_mask = batch_input_ids == self.pad_token

            for t in range(max_target_length):
                current_embeddings = self.embedding_layer(current_tokens)

                # Centralized Decoding Step
                (
                    vocab_distributions,
                    encoder_attention_distributions,
                    decoder_attention_distributions,
                    p_gens,
                    decoder_hidden_states,
                    decoder_cell_states,
                    batch_cummulative_encoder_attention_scores,
                    previous_decoder_hidden_states,
                ) = self._decoder_step(
                    current_embeddings=current_embeddings,
                    decoder_hidden_states=decoder_hidden_states,
                    decoder_cell_states=decoder_cell_states,
                    batch_encoder_hidden_states=batch_encoder_hidden_states,
                    batch_cummulative_encoder_attention_scores=batch_cummulative_encoder_attention_scores,
                    previous_decoder_hidden_states=previous_decoder_hidden_states,
                    input_padding_mask=input_padding_mask,
                    is_first_step=(t == 0),
                )

                # Routing logic based on mode
                if mode == "teacher":
                    next_tokens = batch_target_ids[:, t]
                    current_tokens = self._safe_ids(next_tokens)
                    next_token_probs = p_gens * F.pad(
                        vocab_distributions, (self.end_token, max_num_oovs)
                    ).gather(1, next_tokens.unsqueeze(1)).squeeze(1) + (1 - p_gens) * (
                        encoder_attention_distributions
                        * (batch_input_ids == next_tokens.unsqueeze(1)).float()
                    ).sum(
                        dim=1
                    )

                    nll_losses = nll_losses - (
                        torch.log(next_token_probs + 1e-9).masked_fill(
                            t >= target_lengths, 0
                        )
                    )

                elif mode in ["sampling", "greedy"]:
                    generator_probs = F.pad(
                        p_gens.unsqueeze(1) * vocab_distributions,
                        (self.end_token, max_num_oovs),
                    )
                    pointer_probs = (1 - p_gens).unsqueeze(
                        1
                    ) * encoder_attention_distributions
                    final_distributions = generator_probs.scatter_add(
                        1, batch_input_ids, pointer_probs
                    )

                    if mode == "sampling":
                        next_tokens = torch.multinomial(
                            final_distributions, num_samples=1
                        ).squeeze(1)
                        next_token_log_probs = torch.log(
                            final_distributions[
                                torch.arange(batch_size, device=self.device),
                                next_tokens,
                            ]
                            + 1e-9
                        )
                        cummulative_sampling_log_probs = (
                            cummulative_sampling_log_probs
                            + next_token_log_probs * is_continues
                        )
                    else:  # greedy
                        next_tokens = torch.argmax(final_distributions, dim=1)

                    is_continues = is_continues * (next_tokens != self.end_token)
                    current_tokens = self._safe_ids(next_tokens)

                    sequence_steps = (
                        torch.where(is_continues.bool(), next_tokens, self.end_token)
                    ).unsqueeze(1)

                    if mode == "sampling":
                        sampling_sequences = torch.cat(
                            [sampling_sequences, sequence_steps], dim=1
                        )
                    else:
                        greedy_sequences = torch.cat(
                            [greedy_sequences, sequence_steps], dim=1
                        )

            if mode == "teacher":
                nll_loss = nll_losses.sum()
            elif mode == "sampling":
                sampling_sequence_metrics = []
                for batch_idx, sampling_sequence in enumerate(sampling_sequences):
                    sampling_sequence_metric = compute_metric(
                        "rouge2",
                        token_ids_to_text(
                            self.tokenizer,
                            sampling_sequence,
                            oov_lists[batch_idx],
                        ),
                        (
                            target_texts[batch_idx]
                            if target_texts is not None
                            else token_ids_to_text(
                                self.tokenizer,
                                batch_target_ids[batch_idx],
                                oov_lists[batch_idx],
                            )
                        ),
                    )["rouge2"][0]
                    sampling_sequence_metrics.append(sampling_sequence_metric)
            elif mode == "greedy":
                greedy_sequence_metrics = []
                for batch_idx, greedy_sequence in enumerate(greedy_sequences):
                    greedy_sequence_metric = compute_metric(
                        "rouge2",
                        token_ids_to_text(
                            self.tokenizer,
                            greedy_sequence,
                            oov_lists[batch_idx],
                        ),
                        (
                            target_texts[batch_idx]
                            if target_texts is not None
                            else token_ids_to_text(
                                self.tokenizer,
                                batch_target_ids[batch_idx],
                                oov_lists[batch_idx],
                            )
                        ),
                    )["rouge2"][0]
                    greedy_sequence_metrics.append(greedy_sequence_metric)

        if return_rl_loss:
            return (
                (
                    torch.tensor(greedy_sequence_metrics, device=self.device)
                    - torch.tensor(sampling_sequence_metrics, device=self.device)
                )
                * cummulative_sampling_log_probs
            ).sum()
        else:
            return nll_loss

    def train_one_batch(
        self,
        batch_input_ids: torch.Tensor,
        batch_target_ids: torch.Tensor,
        oov_lists: List[List[str]],
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        target_texts: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, float]:
        self.train()
        self.optimizer.zero_grad()

        is_cuda = self.device.type == "cuda"

        with torch.amp.autocast(device_type="cuda") if is_cuda else nullcontext():
            nll_loss = self.compute_loss(
                batch_input_ids,
                batch_target_ids,
                oov_lists,
                input_lengths,
                target_lengths,
                target_texts,
                return_rl_loss=False,
            )

            if is_cuda:
                self.scaler.scale(nll_loss * self.loss_scale).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                (nll_loss * self.loss_scale).backward()
                self.optimizer.step()

            torch.cuda.empty_cache()

            if self.rl_loss_factor > 0:
                rl_loss = self.compute_loss(
                    batch_input_ids,
                    batch_target_ids,
                    oov_lists,
                    input_lengths,
                    target_lengths,
                    target_texts,
                    return_rl_loss=True,
                )
                if is_cuda:
                    self.scaler.scale(
                        rl_loss * self.loss_scale * self.rl_loss_factor
                    ).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    (rl_loss * self.loss_scale * self.rl_loss_factor).backward()
                    self.optimizer.step()
            else:
                rl_loss = torch.tensor(0.0)

            torch.cuda.empty_cache()

        return tensor_dict_to_scalar(
            {"total_loss": nll_loss + rl_loss, "nll_loss": nll_loss, "rl_loss": rl_loss}
        )

    def validate_one_batch(
        self,
        batch_input_ids: torch.Tensor,
        batch_target_ids: torch.Tensor,
        oov_lists: List[List[str]],
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        target_texts: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            nll_loss = self.compute_loss(
                batch_input_ids,
                batch_target_ids,
                oov_lists,
                input_lengths,
                target_lengths,
                target_texts,
                return_rl_loss=False,
            )
            rl_loss = self.compute_loss(
                batch_input_ids,
                batch_target_ids,
                oov_lists,
                input_lengths,
                target_lengths,
                target_texts,
                return_rl_loss=True,
            )
            return tensor_dict_to_scalar(
                {
                    "total_loss": nll_loss + rl_loss,
                    "nll_loss": nll_loss,
                    "rl_loss": rl_loss,
                }
            )

    def infer(
        self,
        batch_input_ids: torch.Tensor,
        max_output_length: int = 100,
        beam_width: int = 6,
        trigram_penalty: float = -30.0,
        bigram_penalty: float = -15.0,
        unigram_penalty: float = -2.0,
        penalty_range: int = 15,
        return_attention: bool = False,
        return_embedding: bool = False,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            # 1. Prepare Inputs & Beam Search
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
            input_padding_mask = batch_input_ids == self.pad_token

            # 2. Encode Source Sequence & Initialize Decoder States
            batch_embeddings = self._safe_embed(batch_input_ids)
            input_lengths = (batch_input_ids != self.pad_token).sum(dim=1)
            batch_encoder_hidden_states, decoder_hidden_states, decoder_cell_states = (
                self._encode(batch_input_ids, input_lengths)
            )
            batch_cummulative_encoder_attention_scores = torch.zeros(
                batch_size, batch_input_ids.shape[1], device=self.device
            )
            previous_decoder_hidden_states = torch.empty(
                batch_size, 0, self.hidden_dim * 2, device=self.device
            )
            current_tokens = torch.tensor(
                [self.start_token] * batch_size, device=self.device
            )

            if return_attention:
                cross_attention_distributions_list = []
                decoder_attention_distributions_list = []

            # 3. Unified Autoregressive Decoding Loop
            for t in range(max_output_length):
                current_embeddings = self._safe_embed(current_tokens)

                # Centralized Decoding Step
                (
                    vocab_distributions,
                    encoder_attention_distributions,
                    decoder_attention_distributions,
                    p_gens,
                    decoder_hidden_states,
                    decoder_cell_states,
                    batch_cummulative_encoder_attention_scores,
                    previous_decoder_hidden_states,
                ) = self._decoder_step(
                    current_embeddings=current_embeddings,
                    decoder_hidden_states=decoder_hidden_states,
                    decoder_cell_states=decoder_cell_states,
                    batch_encoder_hidden_states=batch_encoder_hidden_states,
                    batch_cummulative_encoder_attention_scores=batch_cummulative_encoder_attention_scores,
                    previous_decoder_hidden_states=previous_decoder_hidden_states,
                    input_padding_mask=input_padding_mask,
                    is_first_step=(t == 0),
                )

                # Final Distribution with Pointer Mechanism
                generator_probs = F.pad(
                    p_gens.unsqueeze(1) * vocab_distributions,
                    (self.end_token, num_oovs),
                )
                pointer_probs = (1 - p_gens).unsqueeze(
                    1
                ) * encoder_attention_distributions
                batch_final_distributions = generator_probs.scatter_add(
                    1, batch_input_ids, pointer_probs
                )

                # Beam Search Routing
                if t == 0:
                    chosen_tokens = beam_search.init_from_first_topk(
                        batch_final_distributions
                    )

                    # Expand states from (Batch) to (Batch * Beam)
                    decoder_hidden_states = decoder_hidden_states.repeat_interleave(
                        beam_width, dim=1
                    )
                    decoder_cell_states = decoder_cell_states.repeat_interleave(
                        beam_width, dim=1
                    )
                    batch_encoder_hidden_states = (
                        batch_encoder_hidden_states.repeat_interleave(beam_width, dim=0)
                    )
                    batch_cummulative_encoder_attention_scores = (
                        batch_cummulative_encoder_attention_scores.repeat_interleave(
                            beam_width, dim=0
                        )
                    )
                    previous_decoder_hidden_states = (
                        previous_decoder_hidden_states.repeat_interleave(
                            beam_width, dim=0
                        )
                    )
                    input_padding_mask = input_padding_mask.repeat_interleave(
                        beam_width, dim=0
                    )
                    batch_input_ids = batch_input_ids.repeat_interleave(
                        beam_width, dim=0
                    )

                    if return_attention:
                        cross_attention_distributions_list.append(
                            encoder_attention_distributions.repeat_interleave(
                                beam_width, dim=0
                            )
                        )
                else:
                    chosen_tokens, chosen_beam_indices = beam_search.advance(
                        batch_final_distributions,
                        trigram_penalty,
                        bigram_penalty,
                        unigram_penalty,
                        penalty_range,
                    )

                    # Select specific beam branches
                    decoder_hidden_states = decoder_hidden_states[
                        :, chosen_beam_indices, :
                    ]
                    decoder_cell_states = decoder_cell_states[:, chosen_beam_indices, :]
                    batch_cummulative_encoder_attention_scores = (
                        batch_cummulative_encoder_attention_scores[chosen_beam_indices]
                    )
                    previous_decoder_hidden_states = previous_decoder_hidden_states[
                        chosen_beam_indices
                    ]
                    batch_input_ids = batch_input_ids[chosen_beam_indices]

                    if return_attention:
                        cross_attention_distributions_list.append(
                            encoder_attention_distributions
                        )
                        decoder_attention_distributions_list.append(
                            decoder_attention_distributions
                        )

                        # Sync historical attention branches
                        for i in range(len(cross_attention_distributions_list)):
                            cross_attention_distributions_list[i] = (
                                cross_attention_distributions_list[i][
                                    chosen_beam_indices
                                ]
                            )
                        for i in range(len(decoder_attention_distributions_list)):
                            decoder_attention_distributions_list[i] = (
                                decoder_attention_distributions_list[i][
                                    chosen_beam_indices
                                ]
                            )

                current_tokens = chosen_tokens
                if beam_search.finishes.all():
                    break

            # 4. Finalize Outputs
            chosen_beam_indices = beam_search.finalize_best_beams()
            output = {"output_ids": beam_search.sequences[chosen_beam_indices, 1:]}

            if return_embedding:
                output["input_embeddings"] = batch_embeddings
            if return_attention:
                output["cross_attention_distributions"] = torch.stack(
                    [
                        attn[chosen_beam_indices]
                        for attn in cross_attention_distributions_list
                    ],
                    dim=1,
                )
                output["decoder_attention_distributions"] = torch.stack(
                    [
                        torch.nn.functional.pad(
                            attn[chosen_beam_indices],
                            (
                                0,
                                len(decoder_attention_distributions_list)
                                - attn.shape[1],
                            ),
                        )
                        for attn in decoder_attention_distributions_list
                    ],
                    dim=1,
                )

            return output
