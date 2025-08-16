import torch
from typing import Any, Dict, List, Optional, Union
from transformers import  WhisperProcessor

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union



def shift_tokens_right(label_ids: torch.Tensor, decoder_start_token_id: int) -> torch.Tensor:
    shifted_label_ids = torch.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id
    return shifted_label_ids

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(
        self,
        processor: Any,
        decoder_start_token_id: int,
        input_padding: Union[bool, str] = "longest",
        target_padding: Union[bool, str] = "longest",
        max_input_length: Optional[float] = None,
        max_target_length: Optional[int] = None,
    ):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
        self.input_padding = input_padding
        self.target_padding = target_padding
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they need different padding methods
        # Extract input features and labels from the features list
        input_features = {"input_features": [feature["input_features"] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}


        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            max_length=self.max_input_length,
            padding=self.input_padding,
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Pad target labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Extract labels and attention mask
        labels = labels_batch["input_ids"]

        # Remove the decoder_start_token_id from labels if present
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
            labels_batch["attention_mask"] = labels_batch["attention_mask"][:, 1:]

        # Shift labels to create decoder input IDs using the helper function
        decoder_input_ids = shift_tokens_right(labels, self.decoder_start_token_id)
        decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)

        # Replace padding in labels with -100 for loss computation
        labels = labels.masked_fill(labels_batch["attention_mask"] == 0, -100)

        # Add labels and decoder input IDs to the batch
        batch["labels"] = labels  # Labels with -100 for ignored tokens
        batch["decoder_input_ids"] = decoder_input_ids  # Shifted decoder input IDs

        llama_logits = [feature["chunk_text_logits"] for feature in features]
        llama_hidden = [feature["context_embeddings"] for feature in features]

        batch["context_embeddings"] = torch.stack(llama_hidden, dim=0)
        batch["chunk_text_logits"] = torch.stack(llama_logits, dim=0)

        return batch

processor = WhisperProcessor.from_pretrained("BayanDuygu/whisper-puncted-timed")


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=50258,
        input_padding="longest",
        target_padding="longest",
        max_input_length=32000,  # Adjust based on your dataset
        max_target_length=128,
)

