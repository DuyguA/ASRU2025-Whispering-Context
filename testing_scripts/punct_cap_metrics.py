from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from commons import decompose_into_punct_tags, is_token_punct, strip_tags, strip_puncts_tags, find_punct


def cap_success(cap_true_labels, cap_pred_labels):
  capitalization_metrics = {
        "true_labels": cap_true_labels,
        "pred_labels": cap_pred_labels,
        "precision": precision_score(cap_true_labels, cap_pred_labels, zero_division=0),
        "recall": recall_score(cap_true_labels, cap_pred_labels, zero_division=0),
        "f1": f1_score(cap_true_labels, cap_pred_labels, zero_division=0),
        "accuracy": accuracy_score(cap_true_labels, cap_pred_labels),
  }
  return capitalization_metrics

def punct_success(all_punct_refs, all_punct_preds):
    punctuation_metrics = {}
    for punct, true_labels in all_punct_refs.items():
        pred_labels = all_punct_preds[punct]
        punctuation_metrics[punct] = {
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "precision": precision_score(true_labels, pred_labels, zero_division=0),
            "recall": recall_score(true_labels, pred_labels, zero_division=0),
            "f1": f1_score(true_labels, pred_labels, zero_division=0),
            "accuracy": accuracy_score(true_labels, pred_labels),
        }
    return punctuation_metrics


def calculate_punct_cap_success(aligned_ref_tokens, aligned_hyp_tokens):
    """
    Calculates metrics (precision, recall, F1, accuracy) for capitalization and punctuation using sklearn.

    Args:
        aligned_ref_tokens (list): List of reference tokens (aligned).
        aligned_hyp_tokens (list): List of hypothesis tokens (aligned).

    Returns:
        dict: Dictionary containing precision, recall, F1, and accuracy for capitalization and punctuation.
    """

    # Initialize labels for capitalization
    cap_true_labels = []  # Ground truth for capitalization
    cap_pred_labels = []  # Predicted values for capitalization

    # Initialize labels for punctuation
    punct_true_labels = {}  # Ground truth for each punctuation type
    punct_pred_labels = {}

    # Compare aligned tokens
    for ref_token, hyp_token in zip(aligned_ref_tokens, aligned_hyp_tokens):
        if ref_token == "#" or hyp_token == "*":  # Skip unaligned tokens
            continue

        ref_token_puncted, hyp_token_puncted = strip_tags(ref_token), strip_tags(hyp_token)
        ref_token_raw, hyp_token_raw = strip_puncts_tags(ref_token), strip_puncts_tags(hyp_token)

        # Capitalization: Compare lowercased version of tokens
        if ref_token_raw.lower() == hyp_token_raw.lower():
            cap_true_labels.append(1)  # Correct capitalization
            cap_pred_labels.append(1 if ref_token_raw == hyp_token_raw else 0)
        else:
            cap_true_labels.append(0)  # Incorrect capitalization
            cap_pred_labels.append(0)

        # Punctuation: Check for punctuation marks in tokens
        ref_punct = find_punct(ref_token)
        hyp_punct = find_punct(hyp_token)

        if ref_punct:
            if ref_punct not in punct_true_labels:
                punct_true_labels[ref_punct] = []
                punct_pred_labels[ref_punct] = []

            punct_true_labels[ref_punct].append(1)  # Ground truth: punctuation exists in reference
            punct_pred_labels[ref_punct].append(1 if ref_punct == hyp_punct else 0)
        elif hyp_punct:
            if hyp_punct not in punct_true_labels:
                punct_true_labels[hyp_punct] = []
                punct_pred_labels[hyp_punct] = []

            punct_true_labels[hyp_punct].append(0)  # Ground truth: punctuation doesn't exist in reference
            punct_pred_labels[hyp_punct].append(1)

    # Calculate metrics for capitalization
    capitalization_metrics = {
        "true_labels": cap_true_labels,
        "pred_labels": cap_pred_labels,
        "precision": precision_score(cap_true_labels, cap_pred_labels, zero_division=0),
        "recall": recall_score(cap_true_labels, cap_pred_labels, zero_division=0),
        "f1": f1_score(cap_true_labels, cap_pred_labels, zero_division=0),
        "accuracy": accuracy_score(cap_true_labels, cap_pred_labels),
    }

    # Calculate metrics for punctuation
    punctuation_metrics = {}
    for punct in punct_true_labels:
        punctuation_metrics[punct] = {
            "true_labels": punct_true_labels[punct],
            "pred_labels": punct_pred_labels[punct],
            "precision": precision_score(punct_true_labels[punct], punct_pred_labels[punct], zero_division=0),
            "recall": recall_score(punct_true_labels[punct], punct_pred_labels[punct], zero_division=0),
            "f1": f1_score(punct_true_labels[punct], punct_pred_labels[punct], zero_division=0),
            "accuracy": accuracy_score(punct_true_labels[punct], punct_pred_labels[punct]),
        }

    return {
        "capitalization": capitalization_metrics,
        "punctuation": punctuation_metrics,
    }

