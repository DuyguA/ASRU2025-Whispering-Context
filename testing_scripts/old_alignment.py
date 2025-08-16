from difflib import SequenceMatcher

def align_texts(reference, hypothesis):
    """
    Aligns reference and hypothesis texts using difflib and marks insertions, deletions, and substitutions.

    Args:
        reference (str): Reference text.
        hypothesis (str): Hypothesis text.

    Returns:
        alignment (dict): Contains aligned reference, aligned hypothesis, and operations sequence.
            - 'aligned_ref': Aligned reference with deletions marked as '#'.
            - 'aligned_hyp': Aligned hypothesis with insertions marked as '*'.
            - 'operations': Sequence of alignment operations ('I', 'D', 'S', '=').
    """
    # Tokenize the reference and hypothesis texts
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, ref_tokens, hyp_tokens)

    aligned_ref = []
    aligned_hyp = []
    operations = []

    #for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    #  print(f"Tag: {tag}, Ref: {ref_tokens[i1:i2]}, Hyp: {hyp_tokens[j1:j2]}")
    #  print("----------------------")

    # Iterate through matching blocks and alignment tags
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Tokens match
            aligned_ref.extend(ref_tokens[i1:i2])
            aligned_hyp.extend(hyp_tokens[j1:j2])
            #print(aligned_hyp, "1")
            operations.extend(["="] * (i2 - i1))
        elif tag == "replace":
            # Tokens are substituted
            aligned_ref.extend(ref_tokens[i1:i2])
            aligned_hyp.extend(hyp_tokens[j1:j2])
            #print(aligned_hyp, "2")
            operations.extend(["S"] * (i2 - i1))  # 'S' for substitution
        elif tag == "delete":
            # Tokens deleted in hypothesis
            aligned_ref.extend(ref_tokens[i1:i2])
            aligned_hyp.extend(["*"] * (i2 - i1))  # '*' for insertion in hypothesis
            #print(aligned_hyp, "3")
            operations.extend(["D"] * (i2 - i1))  # 'D' for deletion
        elif tag == "insert":
            # Tokens inserted in hypothesis
            aligned_ref.extend(["#"] * (j2 - j1))  # '#' for insertion in reference
            aligned_hyp.extend(hyp_tokens[j1:j2])
            #print(aligned_hyp, "4")
            operations.extend(["I"] * (j2 - j1))  # 'I' for insertion

    return aligned_ref, aligned_hyp, operations


def display_alignment_with_wrapping(aligned_ref, aligned_hyp, operations, max_width=80):
    """
    Displays aligned reference and hypothesis in two lines with padding,
    wrapping lines to fit within a maximum width.

    Args:
        aligned_ref (list): Aligned reference tokens with deletions marked as '#'.
        aligned_hyp (list): Aligned hypothesis tokens with insertions marked as '*'.
        operations (list): Sequence of operations ('I', 'D', 'S', '=').
        max_width (int): Maximum width of each line for wrapping.
    """
    # Determine the width of each column (max length of ref and hyp tokens)
    token_lengths = [
        max(len(ref_token or ""), len(hyp_token or ""))
        for ref_token, hyp_token in zip(aligned_ref, aligned_hyp)
    ]

    # Create padded lines
    ref_line = []
    hyp_line = []

    for ref_token, hyp_token, width in zip(aligned_ref, aligned_hyp, token_lengths):
        # Pad reference token
        if ref_token is None:
            ref_line.append("#".ljust(width))  # Mark insertion with '#'
        else:
            ref_line.append(ref_token.ljust(width))

        # Pad hypothesis token
        if hyp_token is None:
            hyp_line.append("*".ljust(width))  # Mark deletion with '*'
        else:
            hyp_line.append(hyp_token.ljust(width))

    # Join the tokens into single strings
    ref_line_str = " ".join(ref_line)
    hyp_line_str = " ".join(hyp_line)

    # Split lines into chunks that fit within max_width
    def split_into_chunks(line, max_width):
        return [line[i:i+max_width] for i in range(0, len(line), max_width)]

    ref_chunks = split_into_chunks(ref_line_str, max_width)
    hyp_chunks = split_into_chunks(hyp_line_str, max_width)

    # Print the aligned strings chunk by chunk
    for ref_chunk, hyp_chunk in zip(ref_chunks, hyp_chunks):
        print("REF:", ref_chunk)
        print("HYP:", hyp_chunk)
        print()  # Add a blank line between chunks
