import heapq

import torch


def beam_search(
    predictor,
    start_state,
    beam_width=3,
    max_state_length=20,
    end_state_indicator=None,
    unrepeated_trigram=False,
):
    beams = [(0.0, start_state)]

    for _ in range(max_state_length):
        candidates = []
        for score, state in beams:
            if end_state_indicator is not None and end_state_indicator(state):
                candidates.append((score, state))
                continue

            state_distribution, new_state = predictor(state)
            log_probs = torch.log(state_distribution + 1e-9)

            for token_idx, log_prob in enumerate(log_probs):
                new_sequence = state["sequence"] + [token_idx]

                if unrepeated_trigram and len(new_sequence) >= 3:
                    trigram = tuple(new_sequence[-3:])
                    existing_trigrams = set(
                        tuple(new_sequence[i : i + 3])
                        for i in range(len(new_sequence) - 3)
                    )
                    if trigram in existing_trigrams:
                        continue

                candidates.append(
                    (
                        score + log_prob.item(),
                        {**new_state, "sequence": new_sequence},
                    )
                )

        beams = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

    return beams[0][1]
