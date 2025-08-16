import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss  # GeomLoss is a library for optimal transport


class OptimalTransportAlign(nn.Module):
    def __init__(self, llama_vocab_size, whisper_vocab_size, projection_dim, sinkhorn_blur=0.05):
        super(OptimalTransportAlign, self).__init__()
        self.llama_vocab_size = llama_vocab_size
        self.whisper_vocab_size = whisper_vocab_size
        self.projection_dim = projection_dim
        self.sinkhorn_blur = sinkhorn_blur

        # Linear projections to map logits to the shared space
        self.llama_projection = nn.Linear(llama_vocab_size, projection_dim)
        self.whisper_projection = nn.Linear(whisper_vocab_size, projection_dim)

        # Sinkhorn OT solver
        self.ot_solver = SamplesLoss("sinkhorn", blur=self.sinkhorn_blur)

        # Optional: Project back to Whisper's vocabulary space if needed
        self.output_projection = nn.Linear(projection_dim, whisper_vocab_size)

    def forward(self, whisper_logits, llama_logits):
        batch_size, whisper_seq_len, whisper_vocab_size = whisper_logits.shape
        _, llama_seq_len, llama_vocab_size = llama_logits.shape

        # Project logits to the shared embedding space
        whisper_embeds = self.whisper_projection(whisper_logits)  # [batch_size, whisper_seq_len, projection_dim]
        llama_embeds = self.llama_projection(llama_logits)        # [batch_size, llama_seq_len, projection_dim]

        # Flatten embeddings for OT computation (batch treated independently)
        whisper_embeds_flat = whisper_embeds.view(batch_size * whisper_seq_len, -1)  # [N, projection_dim]
        llama_embeds_flat = llama_embeds.view(batch_size * llama_seq_len, -1)        # [M, projection_dim]

        # Solve the OT problem to get the transport cost
        transport_cost = self.ot_solver(whisper_embeds_flat, llama_embeds_flat)  # Outputs a scalar cost

        # Align the embeddings using the transport cost (if needed)
        # Note: GeomLoss computes alignment implicitly, so you don't get the explicit transport plan here.

        # Optionally project back to Whisper's vocabulary space
        # Here, aligned embeddings would be equivalent to whisper_embeds (no explicit transport plan generated)
        aligned_llama_logits = self.output_projection(whisper_embeds)

        return aligned_llama_logits, transport_cost

