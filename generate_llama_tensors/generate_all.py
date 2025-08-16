import torch
import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training






model_name = "BayanDuygu/llama1b-rawtext"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"





dataset = load_dataset("BayanDuygu/contexts-paragraphs")


base_model = AutoModelForCausalLM.from_pretrained("BayanDuygu/base_llama1B")
model = PeftModel.from_pretrained(base_model, model_name)


model = model.to("cuda:0")
model.eval()


def process_chunk(chunkjs, paragraph_tokens_embeddings, offset_mapping):
    chunk_no = chunkjs["global_chunk_number"]

    chunk_start, chunk_end = chunkjs["start_index"], chunkjs["end_index"]
    chunk_text_embeddings = chunktext_embeddings(paragraph_token_embeddings, chunk_start, chunk_end, offset_mapping)
    chunk_js = {"chunk_no": chunk_no, "chunk_text_embeddings": chunk_text_embeddings, "context_embeddings": []}

    cindices = chunkjs["context_indices"]
    widths = [64, 128, 256, 512, 1024, 2048, 4096]
    for width, (cstart, cend) in zip(widths, cindices):
      cembedding =  context_embedding(paragraph_token_embeddings, cstart, cend, offset_mapping)
      chunkjs["context_embeddings"].append(cembedding)
    return chunk_js



def generate_labels(text):
  text_labels = tokenizer(text, return_tensors="pt")
  inputs = text_labels.to("cuda:0")
  outputs = model(**inputs, return_dict=True, output_hidden_states=True)
  final_layer_output = outputs.hidden_states[-1].cpu()
  paragraph_embedding = final_layer_output[:, -1, :]
  return paragraph_embedding.tolist()

def generate_token_logits(text):
  text_labels = tokenizer(text, return_tensors="pt")
  inputs = text_labels.to("cuda:0")
  outputs = model(**inputs, return_dict=True, output_logits=True)
  logits = outputs.logits.cpu()
  return logits.tolist()



tensors_dir = "llama_tensors/"

with torch.no_grad():
    for instance in dataset["train"]:
      file_text = instance["file_text"]
      file_tokens = file_text.split()
      chunks = instance["chunks"]

      for chunkjs in chunks:
        chunk_no = chunkjs["global_chunk_number"]
        print(chunk_no)
        if chunk_no in [7, 52, 8268]:
            pass
        else:
            continue

        chunk_text = chunkjs["chunk_text"]
        chunk_text_embeddings =  generate_token_logits(chunk_text)

        filejs = {"chunk_no": chunk_no, "chunk_text_embeddings": chunk_text_embeddings, "context_embeddings": []}

        cindices = chunkjs["context_indices"]
        widths = [64, 128, 256, 512, 1024, 2048, 4096]
        for width, (cstart, cend) in zip(widths, cindices):
            ctext = file_tokens[cstart:cend]
            ctext = " ".join(ctext)
            chid = generate_labels(ctext)
            filejs["context_embeddings"].append(chid)

        filen = tensors_dir + "chunk" + str(chunk_no) + ".json"
        with open(filen, "w") as ofile:
          outjs = json.dumps(filejs)
          ofile.write(outjs+"\n")

