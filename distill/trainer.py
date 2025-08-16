import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import numpy as np

import os, copy, json, csv
from tqdm import tqdm

from models import DistilledWhisper
from custom_dataset import PunctedDataset
from custom_collator import data_collator
from loss_functions import get_loss_class

from torch.utils.data import DataLoader

class Trainer:
  def __init__(self, args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_set = PunctedDataset("train", args.window_size)
    #test_set = PunctedDataset("test", args.window_size)

    self.train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, collate_fn=data_collator)
    #self.valid_loader = DataLoader(test_set, batch_size=args.val_batch_size, shuffle=False, collate_fn=data_collator)
    print('Dataset prep done!\n')

    if torch.cuda.device_count() > 0:
      print(f"{torch.cuda.device_count()} GPUs found")

    print('Initializing model....')
    model = DistilledWhisper()

    model = nn.DataParallel(model)
    model.to(device)
    params = model.parameters()
    self.scaler = GradScaler()

    self.optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    self.device = device
    self.model = model


    self.loss_class = get_loss_class(args.loss_type, args.alpha, args.beta, args.temperature)
    self.mode = args.loss_type


    self.args = args
    self.epoch_accuracies = []
    self.all_losses = []

  def train(self):
    best_epoch = 0
    print("First epoch will start soon")
    for epoch in range(self.args.epochs):
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
      loss = self.train_epoch()
      #eval_loss  = self.eval()
      #print(f'Eval loss: {eval_loss:.3f}')
    self.model.module.whisper.save_pretrained(args.output_dir)

  def train_epoch(self):
    self.model.train()
    epoch_loss = 0
    loader = self.train_loader
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        self.optimizer.zero_grad()
        labels = batch["labels"].to(self.device)
        features = batch["input_features"].to(self.device)  # target
        decoder_ids = batch["decoder_input_ids"].to(self.device)
        llama_logits = batch["chunk_text_logits"].to(self.device)
        llama_hidden = batch["context_embeddings"].to(self.device)
        #print(llama_hidden.shape, "hidden shape")
        #print(llama_logits.shape, "logits shape")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
          whisper_logits, whisper_hidden, llama_logits, llama_hidden, transport_cost = self.model(features, decoder_ids, llama_logits, llama_hidden)
          #print(student_logits.shape, student_hidden.shape, teacher_hidden, "output shapes")
          loss = self.loss_class(labels, whisper_logits, llama_logits, whisper_hidden, llama_hidden, transport_cost)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        interval = max(len(loader) // 20, 1)
        if i % interval == 0 or i == len(loader) - 1:
            lloss = round(loss.item(), 3)
            print(f'Batch: {i + 1}/{len(loader)}\ttotal loss: {lloss:.3f}')
            self.all_losses.append(lloss) # append epoch losses
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


  def eval(self):
    self.model.eval()
    label_pred = []
    label_true = []

    loader = self.valid_loader 
    total_loss=0

    with torch.no_grad():
      for i, batch in enumerate(loader):
        labels = batch["labels"].to(self.device)
        features = batch["input_features"].to(self.device)
        decoder_ids = batch["decoder_input_ids"].to(self.device)
        llama_logits = batch["chunk_text_logits"].to(self.device)
        llama_hidden = batch["context_embeddings"].to(self.device)

        whisper_logits, whisper_hidden, llama_logits, llama_hidden, trans_cost = self.model(features, decoder_ids, llama_logits, llama_hidden)
        loss = self.loss_class(labels, whisper_logits, llama_logits, whisper_hidden, llama_hidden, trans_cost)

        total_loss += loss.item()
    return total_loss / len(loader)

      


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=48, help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size of testing')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="exp", help='Directory for saving the model')
    parser.add_argument('--teacher_dim', type=int, default="2048", help='Dimensionality of teacher')
    parser.add_argument('--temperature', type=float, default="2.0", help='Temperature for smoothening logits')
    parser.add_argument('--loss_type', type=str, default="logits", help='Loss type for distillation. Options are logits or CLS')
    parser.add_argument('--alpha', type=float, default="0.5", help='Distillation coeff of logits, must be between 0 and 1. Must be nonzero if logits or logits+CLS chosen')
    parser.add_argument('--beta', type=float, default="0", help='Distillation coeff of CLS, must be between 0 and 1. Must be nonzero if CLS is chosen.')
    parser.add_argument('--window_size', type=int, default="64", help='Window length around the chunk text')
    args = parser.parse_args()

    print(args)
    engine = Trainer(args)
    engine.train()

