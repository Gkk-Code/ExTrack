"""Chronological same-identity sampling, using original frame numbers if available."""

import random


def sample_temporal_ids(visible, frame_numbers, history_length, num_templates, max_gap):
    valid = [i for i, flag in enumerate(visible) if bool(flag)]
    if not valid:
        raise ValueError("Trajectory has no usable observations")
    numbers = list(range(len(visible))) if frame_numbers is None else [int(x) for x in frame_numbers]
    if any(b <= a for a, b in zip(numbers, numbers[1:])):
        raise ValueError("Frame numbers must be strictly increasing within an identity")
    # A one-observation trajectory remains a valid appearance-only sample.
    current = random.choice(valid[1:] or valid)
    previous = [i for i in valid if i < current and 0 < numbers[current] - numbers[i] <= max_gap]
    templates = random.choices(previous or [current], k=num_templates)
    history = previous[-history_length:] if history_length else []
    padding = history_length - len(history)
    return sorted(templates), [current], [current] * padding + history, [False] * padding + [True] * len(history)
