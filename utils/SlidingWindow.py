def sliding_window_tokens(
    input_ids,
    window_size=512,
    stride=256,
    pad_token_id=0
):
    windows = []

    for start in range(0, len(input_ids), stride):
        window = input_ids[start:start + window_size]

        if len(window) == 0:
            break

        # Padding window cuối
        if len(window) < window_size:
            window = window + [pad_token_id] * (
                window_size - len(window)
            )

        windows.append(window)

        if start + window_size >= len(input_ids):
            break

    return windows