import torch
from torch import nn
from transformers import WhisperForConditionalGeneration
from optimal_transport import OptimalTransportAlign



class DistilledWhisper(nn.Module):
  def __init__(self, llama_hidden_dim=2048):
    super(DistilledWhisper, self).__init__()
    model = WhisperForConditionalGeneration.from_pretrained("BayanDuygu/whisper-puncted-timed")
    for param in model.model.encoder.parameters():
      param.requires_grad = False

    for param in model.model.encoder.layers[-2:].parameters():
      param.requires_grad = True

    self.whisper = model

    whisper_hidden_dim = self.whisper.config.hidden_size
    self.projection_layer = nn.Linear(llama_hidden_dim, whisper_hidden_dim)

    self.ot_align = OptimalTransportAlign(
      llama_vocab_size=128298,  # Example vocab size for LLaMA
      whisper_vocab_size=51907,  # Example vocab size for Whisper
      projection_dim=128,  # Shared embedding space dimension
      sinkhorn_blur=0.05  # Regularization parameter for Sinkhorn
    )

  def forward(self, features, decoder_input_ids, llama_logits, teacher_hidden=None):
    outputs = self.whisper(input_features=features, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
    whisper_logits = outputs.logits
    decoder_hidden_states = outputs.decoder_hidden_states
    decoder_final = decoder_hidden_states[-1]
    whisper_hidden = decoder_final[:, -1, :]

    projected_teacher_hidden = None
    if teacher_hidden is not None:
      projected_llama_hidden = self.projection_layer(teacher_hidden)
    else:
      projected_llama_hidden = None

    aligned_llama_logits, transport_cost = self.ot_align(whisper_logits, llama_logits)
    return whisper_logits, whisper_hidden, aligned_llama_logits, projected_llama_hidden, transport_cost

