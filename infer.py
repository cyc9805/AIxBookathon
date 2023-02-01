from typing import Optional
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
import argparse
import os

def load_model(model_path):
    model = AutoModelWithLMHead.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(sequence, model_name, max_length, sample):
    model_path = "/home/cyc/bookathon/model"
    model_path = os.path.join(model_path, model_name)
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    if sample:

        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            repetition_penalty=2.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=10
        )

    else:

        final_outputs = model.generate(
            ids,
            do_sample=False,
            beam_search=5,
            max_length=max_length,
            repetition_penalty=2.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
            num_return_sequences=10
        )

    for i, final_output in enumerate(final_outputs):
        print("{}: {}\n".format(i, tokenizer.decode(final_output, skip_special_tokens=True)))

def main():
    parser = argparse.ArgumentParser(description='Infering')
    parser.add_argument('--sequence', default='길을 걷다보면', help='prompt')
    parser.add_argument('--model_name', default='model1', help='name of the fine-tuned model')
    parser.add_argument('--maxlen', default=300, help='max length of the sequence')
    parser.add_argument('--sample', default=True, help='whether to sample or not')
    
    opt = parser.parse_args()

    model_name = opt.model_name
    sequence = opt.sequence
    max_len = opt.maxlen
    sample = opt.sample

    generate_text(sequence, model_name, max_len, sample)

if __name__ == '__main__':
    exit(main())

