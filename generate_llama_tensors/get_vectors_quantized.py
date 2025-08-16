import torch
import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training






model_name = "BayanDuygu/llama1b-rawtext"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"





dataset = load_dataset("BayanDuygu/contexts-paragraphs", split="train")


base_model = AutoModelForCausalLM.from_pretrained("BayanDuygu/base_llama1B")
peft_model = PeftModel.from_pretrained(base_model, model_name)

model = peft_model.merge_and_unload()  # Merges PEFT weights into the base model


model = model.to("cuda:0")
model.eval()


def generate_token_logits_batch(text_list, max_length=128):
    text_labels = tokenizer(
        text_list,
        return_tensors="pt",
        padding="max_length",  # Pad to `max_length`
        truncation=True,       # Truncate sequences longer than `max_length`
        max_length=max_length
    )

    # Move inputs to GPU
    inputs = {key: value.to("cuda:0") for key, value in text_labels.items()}

    # Forward pass through the model
    outputs = model(**inputs, return_dict=True, output_logits=True)

    # Collect logits and move them to CPU
    logits = outputs.logits.cpu()

    # Convert logits to lists and return
    logits = logits.half()
    return logits


# Directory to save tensors
tensors_dir = "llama_tensors/"

BATCH_SIZE = 32
MAX_LENGTH = 32

# Flatten chunks from all instances into a global list
all_chunks = []
for instance in dataset:
    for chunk in instance["chunks"]:
        # Add each chunk with its metadata
        all_chunks.append({
            "chunk_text": chunk["chunk_text"],
            "global_chunk_number": chunk["global_chunk_number"]
        })

print(f"Total chunks available: {len(all_chunks)}")

#all_chunks = all_chunks[:12000]

# Process chunks in batches
with torch.no_grad():
    for i in range(0, len(all_chunks), BATCH_SIZE):
        # Prepare a batch of chunks
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_texts = [chunk["chunk_text"] for chunk in batch_chunks]

        # Generate token logits for the batch
        batch_logits = generate_token_logits_batch(batch_texts, max_length=MAX_LENGTH)

        # Save each chunk's logits to its file
        for j, chunk in enumerate(batch_chunks):
            chunk_no = chunk["global_chunk_number"]
            print(f"Processing and saving logits for chunk {chunk_no}")

            # Save logits to a file (using the global chunk number)
            filen = tensors_dir + f"logits{chunk_no}.json"
            token_logits = batch_logits[j]
            int4_min, int4_max = -8, 7  # Range for signed int4
            scale_factors = token_logits.abs().max(dim=1, keepdim=True).values / int4_max  # Per-vector scale factors
            quantized_logits = torch.round(token_logits / scale_factors).clip(int4_min, int4_max).to(torch.int8)

            newjs  = {'logits': quantized_logits.tolist(), 'scale_factors': scale_factors.tolist()}
            with open(filen, "w") as ofile:
                ofile.write(json.dumps(newjs))

