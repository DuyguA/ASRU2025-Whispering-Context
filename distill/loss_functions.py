import torch.nn as nn
import torch
import torch.nn.functional as F

def calculate_label_loss(logits, labels, smoothing=0.1, ignore_index=-100):
    vocab_size = logits.size(-1)
    confidence = 1.0 - smoothing
    smoothed_labels = torch.full(size=(logits.size(0), logits.size(1), vocab_size), fill_value=smoothing / (vocab_size - 1)).to(logits.device)

    smoothed_labels.scatter_(2, labels.unsqueeze(-1), confidence)
    mask = labels == ignore_index
    smoothed_labels[mask] = 0

    log_probs = nn.functional.log_softmax(logits, dim=-1)
    loss = -torch.sum(log_probs * smoothed_labels, dim=-1)

    loss = loss.masked_fill(mask, 0)
    return loss.sum() / (~mask).sum()  # Average over non-ignored tokens


def calculate_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    # Softmax with temperature for teacher and student logits
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL Divergence (distillation loss)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return kl_loss



def calculate_representation_loss(teacher_hidden, student_hidden):
    # Representation alignment loss (MSE)
    #teacher_hidden = teacher_hidden.squeeze(1)
    teacher_hidden = F.normalize(teacher_hidden, p=2, dim=1)
    student_hidden = F.normalize(student_hidden, p=2, dim=1)

    return F.mse_loss(student_hidden, teacher_hidden)



class CombinedLabelLoss(nn.Module):
    """
    Custom loss function for combining cross-entropy loss with distillation loss.

    Args:
    - temperature (float): Temperature for distillation loss.
    - alpha (float): Weighting factor for distillation loss (0 <= alpha <= 1).
    """
    def __init__(self, alpha, temperature):
        super(CombinedLabelLoss, self).__init__()
        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, labels, student_logits, teacher_logits, student_hidden, teacher_hidden, transport_cost):
        """
        Compute the combined loss.

        Args:
        - student_logits (torch.Tensor): Logits output by the student model. Shape: [batch_size, num_classes]
        - teacher_logits (torch.Tensor): Logits output by the teacher model. Shape: [batch_size, num_classes]
        - labels (torch.Tensor): Ground truth labels. Shape: [batch_size]

        Returns:
        - total_loss (torch.Tensor): Combined loss value.
        """
        label_loss = calculate_label_loss(student_logits, labels)
        distill_loss = calculate_distillation_loss(student_logits, teacher_logits, self.temperature)
        total_loss = self.alpha * distill_loss + 0.001 * transport_cost + (1 - self.alpha - 0.001) * label_loss
        return total_loss

class CombinedReprLoss(nn.Module):
    """
    Custom loss function for combining cross-entropy loss with distillation loss.

    Args:
    - temperature (float): Temperature for distillation loss.
    - alpha (float): Weighting factor for distillation loss (0 <= alpha <= 1).
    """
    def __init__(self, alpha, beta, temperature):
        super(CombinedReprLoss, self).__init__()
        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= beta <= 1, "Beta must be between 0 and 1"
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(self, labels, student_logits, teacher_logits, student_hidden, teacher_hidden, transport_cost):
        label_loss = calculate_label_loss(student_logits, labels)

        rep_loss = calculate_representation_loss(teacher_hidden, student_hidden)

        distill_loss = calculate_distillation_loss(student_logits, teacher_logits, self.temperature)

        total_loss = self.alpha * distill_loss + self.beta * rep_loss + 0.001 * transport_cost + (1 - self.alpha - self.beta - 0.001) * label_loss
        return total_loss


def get_loss_class(loss_type, alpha, beta, temperature):
  if loss_type == "logits":
     loss_obj = CombinedLabelLoss(alpha, temperature)
  elif loss_type == "CLS":
     loss_obj = CombinedReprLoss(alpha, beta, temperature)
  return loss_obj


