import numpy as np
import hashlib

def partition_vocabulary(prev_token, vocab_size, gamma):
    """
    Partitions the vocabulary into green and red lists based on the hash of the given token.

    Parameters:
    - prev_token (int): The previous token s(t-1).
    - vocab_size (int): The size of the vocabulary |V|.
    - gamma (float): The fraction of the vocabulary in the green list (0 < gamma < 1).

    Returns:
    - tuple: (green_list, red_list), both are sets of token indices.
    """
    token_str = str(prev_token)
    hash_digest = hashlib.sha256(token_str.encode()).hexdigest()

    seed = int(hash_digest, 16) % (2**32)
    rng = np.random.RandomState(seed)

    vocab_indices = np.arange(vocab_size)
    rng.shuffle(vocab_indices)

    # Determine the size of the green list
    green_size = int(np.floor(gamma * vocab_size))

    # Split the vocabulary into green and red lists
    green_list = set(vocab_indices[:green_size])
    red_list = set(vocab_indices[green_size:])

    return green_list, red_list

def insert_watermark_softred(initial_tokens, L_t_list, gamma, delta):
    """
    Inserts a watermark into the text by adjusting the logits of the green list tokens.

    Parameters:
    - initial_tokens (list): The initial sequence of tokens.
    - L_t_list (list): A list of logits vectors L_t for each position t.
    - gamma (float): Fraction of vocabulary in the green list (0 < gamma < 1).
    - delta (float): Hardness parameter δ > 0.

    Returns:
    - list: The sequence of tokens with the watermark inserted.
    """
    tokens = initial_tokens.copy()
    T = len(L_t_list)  # Total number of tokens to generate
    K = len(L_t_list[0])  # Vocabulary size

    for t in range(T):
        L_t = L_t_list[t]

        prev_token = tokens[-1] if len(tokens) > 0 else 0

        green_list, _ = partition_vocabulary(prev_token, K, gamma)

        adjusted_logits = np.copy(L_t)
        adjusted_logits[list(green_list)] += delta

        exp_logits = np.exp(adjusted_logits)
        p_t = exp_logits / np.sum(exp_logits)

        # Sample the next token s(t) from p_t
        i_t = np.random.choice(K, p=p_t)
        tokens.append(i_t)

    return tokens

def detect_watermark_softred(tokens, vocab_size, gamma):
    """
    Detects the watermark in a sequence of tokens by computing the z-statistic.

    Parameters:
    - tokens (list): The sequence of tokens to analyze.
    - vocab_size (int): The size of the vocabulary |V|.
    - gamma (float): Fraction of vocabulary in the green list (0 < gamma < 1).

    Returns:
    - float: The z-score indicating the likelihood the text is watermarked.
    - int: The number of tokens in the green list.
    - int: The total number of tokens analyzed.
    """
    T = len(tokens) - 1 
    green_token_count = 0

    for t in range(1, len(tokens)):
        prev_token = tokens[t - 1]
        curr_token = tokens[t]

        green_list, _ = partition_vocabulary(prev_token, vocab_size, gamma)

        if curr_token in green_list:
            green_token_count += 1

    expected_green_tokens = gamma * T
    variance = gamma * (1 - gamma) * T
    if variance == 0:
        z = 0
    else:
        z = (green_token_count - expected_green_tokens) / np.sqrt(variance)

    return z, green_token_count, T

def generate_L_t_list(T, K):
    """
    Generates a list of random logits vectors L_t for testing.

    Parameters:
    - T (int): Number of tokens to generate.
    - K (int): Vocabulary size.

    Returns:
    - list: A list of logits vectors L_t.
    """
    L_t_list = []
    for _ in range(T):
        # Generate random logits (simulating a language model's output)
        logits = np.random.randn(K)
        L_t_list.append(logits)
    return L_t_list

def generate_non_watermarked_text_from_logits(initial_tokens, L_t_list):
    """
    Generates non-watermarked text by sampling tokens according to the original logits.

    Parameters:
    - initial_tokens (list): The initial sequence of tokens.
    - L_t_list (list): A list of logits vectors L_t for each position t.

    Returns:
    - list: The sequence of tokens generated without watermarking.
    """
    tokens = initial_tokens.copy()
    T = len(L_t_list)
    K = len(L_t_list[0])

    for t in range(T):
        L_t = L_t_list[t]

        # Apply softmax to get the probability distribution
        exp_logits = np.exp(L_t)
        p_t = exp_logits / np.sum(exp_logits)

        # Sample the next token s(t) from p_t
        i_t = np.random.choice(K, p=p_t)
        tokens.append(i_t)

    return tokens

def example():
    """
    Example demonstrating watermark insertion and detection using the soft red list.
    """
    # Parameters
    vocabulary_size = 1000
    text_length = 500
    initial_tokens = [0]
    gamma = 0.5  # Fraction of vocabulary in the green list
    delta = 2.0  # Hardness parameter δ

    # Generate logits vectors L_t for each position t
    L_t_list = generate_L_t_list(text_length, vocabulary_size)

    watermarked_tokens = insert_watermark_softred(initial_tokens, L_t_list, gamma, delta)

    non_watermarked_tokens = generate_non_watermarked_text_from_logits(initial_tokens, L_t_list)

    z_wm, green_count_wm, T_wm = detect_watermark_softred(watermarked_tokens, vocabulary_size, gamma)
    print("Watermarked text z-score: {:.3f}, Green tokens: {}/{}".format(z_wm, green_count_wm, T_wm))

    z_non_wm, green_count_non_wm, T_non_wm = detect_watermark_softred(non_watermarked_tokens, vocabulary_size, gamma)
    print("Non-watermarked text z-score: {:.3f}, Green tokens: {}/{}".format(z_non_wm, green_count_non_wm, T_non_wm))

if __name__ == "__main__":
    example()
