import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("BayanDuygu/text-chunks")

# 1. Load the LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your model path if local
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


tags_file = "all_tags_puncted.txt"
with open(tags_file, "r", encoding="utf-8") as f:
    new_tokens = [line.strip() for line in f if line.strip()]

print(new_tokens)
tags_file = "all_tags_puncted.txt"


print(f"Adding {len(new_tokens)} new tokens to the tokenizer...")
tokenizer.add_tokens(new_tokens)



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, use_cache=False, device_map="auto")
model.resize_token_embeddings(len(tokenizer))
model.push_to_hub("BayanDuygu/base_llama3B")

# 2. Configure QLoRA with PEFT (Parameter-Efficient Fine-Tuning)
lora_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.01,  # Dropout for regularization
    task_type="CAUSAL_LM",  # Task type for causal language modeling
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


'''
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    return { "text" : [example + EOS_TOKEN for example in examples["text"]] }


for instance in dataset["train"]:
    print(instance)
    break
'''


from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="llama3b-rawtext",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=100,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=True,
    report_to="none",
    push_to_hub_organization="BayanDuygu",
    bf16=True,
    dataset_text_field="text",
    max_seq_length=512,
)


trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    #formatting_func=formatting_prompts_func,
)

# train model
trainer.train()

# 8. Save the fine-tuned model and tokenizer
tokenizer.push_to_hub("BayanDuygu/tokenizer-llama3b-rawtext")


