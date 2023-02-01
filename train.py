from typing import Optional
import argparse
import torch.nn as nn
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling, TextDataset
from transformers import Trainer, TrainingArguments
import os

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

def train(train_file_path,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):

  tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 
  
  train_dataset = load_dataset(train_file_path, tokenizer)
  data_collator = load_data_collator(tokenizer)

  model = AutoModelWithLMHead.from_pretrained("skt/kogpt2-base-v2")
  device = "cuda"
  model.to(device)

  tokenizer.save_pretrained(output_dir)
  model.save_pretrained(output_dir)

  training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
      )

  trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
  )

  model = nn.DataParallel(model)
  trainer.train()
  trainer.save_model()


def main():
    train_data_dir = '/home/cyc/bookathon/data'
    output_dir = '/home/cyc/bookathon/model'

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_name', default='collected_data.txt', type=str, help='name of the data')
    parser.add_argument('--model_name', default='model1', type=str, help='name of the output model')
    parser.add_argument('--batchsize', default=8, type=int, help='batch size')
    parser.add_argument('--epoch', default=5, type=int, help='epoch')
    parser.add_argument('--save_steps', default=500, type=int, help='save steps')
    parser.add_argument('--overwrite_output_dir', default=True, type=bool, help='overwrite_output_dir')

    opt = parser.parse_args()

    data_name = opt.data_name
    model_name = opt.model_name
    overwrite_output_dir = opt.overwrite_output_dir
    per_device_train_batch_size = opt.batchsize
    num_train_epochs = opt.epoch
    save_steps = opt.save_steps
    train_file_path = os.path.join(train_data_dir, data_name)
    output_dir = os.path.join(output_dir, model_name)
    
    train(
        train_file_path=train_file_path,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
        )


if __name__ == '__main__':
    exit(main())