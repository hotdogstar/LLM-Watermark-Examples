import numpy as np
import hashlib

def pseudorandom_function(context, candidate_token):
    """
    Pseudorandom function f that maps the latest c tokens and candidate token i to r_{t,i} in [0,1].

    Parameters:
    - context (list): The list of the latest c tokens.
    - candidate_token (int): The candidate token index.

    Returns:
    - float: A pseudorandom number in [0,1).
    """
    combined_input = ' '.join(map(str, context)) + ' ' + str(candidate_token)
    hash_digest = hashlib.sha256(combined_input.encode()).hexdigest()
    hash_int = int(hash_digest, 16)
    r_ti = (hash_int % 10**8) / 10**8
    return r_ti

def insert_watermark(initial_tokens, D_t_list, context_length):
    """
    Watermark insertion function that chooses tokens to maximize r_{t,i}^{1/p_{t,i}}.

    Parameters:
    - initial_tokens (list): The initial sequence of tokens.
    - D_t_list (list): A list of probability distributions D_t for each position t.
    - context_length (int): The length of the context c to consider.

    Returns:
    - list: The sequence of tokens with the watermark inserted.
    """
    tokens = initial_tokens.copy()
    T = len(D_t_list)
    K = len(D_t_list[0])

    for t in range(T):
        D_t = D_t_list[t]  # Probability distribution at position t

        # Get the context for the pseudorandom function
        context = tokens[-context_length:] if len(tokens) >= context_length else tokens

        s_t = []  # List to store s_{t,i} values for each candidate token i

        for i in range(K):
            r_ti = pseudorandom_function(context, i)
            # Compute s_{t,i} = r_{t,i}^{1/p_{t,i}}
            s_ti = r_ti ** (1 / D_t[i])
            s_t.append(s_ti)

        # Choose the token i that maximizes s_{t,i}
        i_t = np.argmax(s_t)
        tokens.append(i_t)

    return tokens

def detect_watermark(tokens, context_length):
    """
    Watermark detection function that computes a z-score to assess the probability the text is watermarked.

    Parameters:
    - tokens (list): The sequence of tokens to analyze.
    - context_length (int): The length of the context c used during watermark insertion.

    Returns:
    - tuple: z-score and mean of r_{t,i_t} values.
    """
    r_values = []

    for t in range(context_length, len(tokens)):
        context = tokens[t - context_length:t]
        i_t = tokens[t]
        r_ti = pseudorandom_function(context, i_t)
        r_values.append(r_ti)

    r_array = np.array(r_values)
    T = len(r_array)
    sample_mean = np.mean(r_array)
    expected_mean = 0.5
    expected_std = np.sqrt(1 / 12)
    standard_error = expected_std / np.sqrt(T)
    z_score = (sample_mean - expected_mean) / standard_error

    return z_score, sample_mean

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
    context_length = 5
    vocabulary_size = 10 
    text_length = 100
    initial_tokens = [0] * context_length 

    D_t_list = generate_D_t_list(text_length, vocabulary_size)

    watermarked_tokens = insert_watermark(initial_tokens, D_t_list, context_length)
    non_watermarked_tokens = generate_non_watermarked_text(initial_tokens, D_t_list)

    z_wm, mean_wm = detect_watermark(watermarked_tokens, context_length)
    print("Watermarked text z-score: {:.3f}, Mean r_ti: {:.3f}".format(z_wm, mean_wm))

    z_non_wm, mean_non_wm = detect_watermark(non_watermarked_tokens, context_length)
    print("Non-watermarked text z-score: {:.3f}, Mean r_ti: {:.3f}".format(z_non_wm, mean_non_wm))

if __name__ == "__main__":
    example()
