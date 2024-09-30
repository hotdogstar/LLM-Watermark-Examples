import numpy as np
import hashlib

def partition_vocabulary(token, vocab_size):
    """
    Partitions the vocabulary into green and red lists based on the hash of the given token.

    Parameters:
    - token (int): The previous token s(t-1).
    - vocab_size (int): The size of the vocabulary K.

    Returns:
    - tuple: (green_list, red_list), both are sets of token indices.
    """
    token_str = str(token)
    hash_digest = hashlib.sha256(token_str.encode()).hexdigest()

    seed = int(hash_digest, 16) % (2**32)
    rng = np.random.RandomState(seed)

    vocab_indices = np.arange(vocab_size)
    rng.shuffle(vocab_indices)

    # Split the vocabulary into green and red lists
    half = vocab_size // 2
    green_list = set(vocab_indices[:half])
    red_list = set(vocab_indices[half:])

    return green_list, red_list

def insert_watermark(initial_tokens, D_t_list):
    """
    Inserts a watermark into the text by sampling tokens only from the green list at each position.

    Parameters:
    - initial_tokens (list): The initial sequence of tokens.
    - D_t_list (list): A list of probability distributions D_t for each position t.

    Returns:
    - list: The sequence of tokens with the watermark inserted.
    """
    tokens = initial_tokens.copy()
    T = len(D_t_list)  # Total number of tokens to generate
    K = len(D_t_list[0])  # Vocabulary size

    for t in range(T):
        D_t = D_t_list[t]  # Probability distribution at position t

        prev_token = tokens[-1] if len(tokens) > 0 else 0

        green_list, _ = partition_vocabulary(prev_token, K)

        D_t_mod = np.copy(D_t)
        D_t_mod = np.where(np.isin(np.arange(K), list(green_list)), D_t_mod, 0.0)

        # Normalize the modified probability distribution
        total_prob = np.sum(D_t_mod)
        if total_prob == 0.0:
            D_t_mod = np.array([1.0 if i in green_list else 0.0 for i in range(K)], dtype=np.float64)
            total_prob = np.sum(D_t_mod)
        D_t_mod /= total_prob

        i_t = np.random.choice(K, p=D_t_mod)
        tokens.append(i_t)

    return tokens

def detect_watermark(tokens, vocab_size):
    """
    Detects the watermark in a sequence of tokens by computing the z-statistic.

    Parameters:
    - tokens (list): The sequence of tokens to analyze.
    - vocab_size (int): The size of the vocabulary K.

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

        # Recompute the green list
        green_list, _ = partition_vocabulary(prev_token, vocab_size)

        if curr_token in green_list:
            green_token_count += 1

    expected_green_tokens = T / 2
    variance = T / 4
    z = (green_token_count - expected_green_tokens) / np.sqrt(variance)
    z *= 2

    return z, green_token_count, T

def generate_D_t_list(T, K):
    """
    Generates a list of random probability distributions D_t for testing.

    Parameters:
    - T (int): Number of tokens to generate.
    - K (int): Vocabulary size.

    Returns:
    - list: A list of probability distributions D_t.
    """
    D_t_list = []

    for _ in range(T):
        # Generate a random probability distribution using the Dirichlet distribution
        p = np.random.dirichlet(np.ones(K))
        D_t_list.append(p)

    return D_t_list

def generate_non_watermarked_text(initial_tokens, D_t_list):
    """
    Generates non-watermarked text by sampling tokens according to D_t.

    Parameters:
    - initial_tokens (list): The initial sequence of tokens.
    - D_t_list (list): A list of probability distributions D_t for each position t.

    Returns:
    - list: The sequence of tokens generated without watermarking.
    """
    tokens = initial_tokens.copy()
    T = len(D_t_list)

    for t in range(T):
        D_t = D_t_list[t]
        # Randomly choose a token according to the probability distribution D_t
        i_t = np.random.choice(len(D_t), p=D_t)
        tokens.append(i_t)

    return tokens

def example():
    """
    Example demonstrating watermark insertion and detection.
    """
    # Parameters
    vocabulary_size = 1000
    text_length = 500
    initial_tokens = [0]

    # Generate probability distributions D_t for each position t
    D_t_list = generate_D_t_list(text_length, vocabulary_size)

    watermarked_tokens = insert_watermark(initial_tokens, D_t_list)

    non_watermarked_tokens = generate_non_watermarked_text(initial_tokens, D_t_list)

    z_wm, green_count_wm, T_wm = detect_watermark(watermarked_tokens, vocabulary_size)
    print("Watermarked text z-score: {:.3f}, Green tokens: {}/{}".format(z_wm, green_count_wm, T_wm))

    z_non_wm, green_count_non_wm, T_non_wm = detect_watermark(non_watermarked_tokens, vocabulary_size)
    print("Non-watermarked text z-score: {:.3f}, Green tokens: {}/{}".format(z_non_wm, green_count_non_wm, T_non_wm))

if __name__ == "__main__":
    example()
