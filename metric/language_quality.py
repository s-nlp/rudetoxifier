import torch
from tqdm import tqdm
import numpy as np

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

class Args:
    def __init__(self):
        self.model_type = 'gpt2'
        self.model_name_or_path = 'sberbank-ai/rugpt2large'

        self.prompt = ''
        self.length = 50
        self.stop_token = '</s>'

        self.k = 5
        self.p = .95
        self.temperature = 1

        self.repetition_penalty = 1
        self.num_return_sequences = 1

        self.device = 'cuda:0'
        self.seed = 42


def get_gpt2_ppl_corpus(test_sentences):
    args = Args()
    args.model_name_or_path = 'sberbank-ai/rugpt2large'
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    lls = []
    weights = []
    for text in tqdm(test_sentences):
        encodings = tokenizer(f'\n{text}\n', return_tensors='pt')
        input_ids = encodings.input_ids.to(args.device)
        target_ids = input_ids.clone()

        w = max(0, len(input_ids[0]) - 1)
        if w > 0:
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs[0]
                ll = log_likelihood.item()
        else:
            ll = 0
        lls.append(ll)
        weights.append(w)

    return np.dot(lls, weights) / sum(weights)