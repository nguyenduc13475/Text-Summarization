import heapq

import torch


def beam_search(
    predictor,
    start_state,
    beam_width=3,
    max_state_length=20,
    end_state_indicator=None,
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
                candidates.append(
                    (
                        score + log_prob,
                        {**new_state, "sequence": state["sequence"] + [token_idx]},
                    )
                )

        beams = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

    return beams[0][1]
