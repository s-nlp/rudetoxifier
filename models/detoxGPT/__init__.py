from tqdm import tqdm
import re
import os

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
        self.model_name_or_path = 'sberbank-ai/rugpt3large_based_on_gpt2'

        self.prompt = ''
        self.length = 50
        self.stop_token = '</s>'

        self.k = 5
        self.p = .95
        self.temperature = 1

        self.repetition_penalty = 1
        self.num_return_sequences = 1

        self.seed=42

        
class detoxGPT:

    def __init__(self, device='cuda', model_path='rugpt3_large_200'):
        self.args = Args()
        self.args.device = device
        
        self.args.model_name_or_path = model_path
        if not os.path.isdir(self.args.model_name_or_path):
            print('Loading pre-trained weights.')
            os.system('gdown --id 1RYUku5_MWXZF2xlIpOTZmi9_DH-SG0lz && mkdir rugpt3_large_200 && unzip rugpt3_large_200.zip -d ' + self.args.model_name_or_path)
        
        model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(self.args.model_name_or_path)
        self.model = model_class.from_pretrained(self.args.model_name_or_path)
        self.model.to(self.args.device)
        
    def generate_sequences(self, prompt_text, delimiter='>>>'):
        self.args.prompt_text = prompt_text


        if prompt_text.endswith('.txt'):
            with open(prompt_text, 'r') as f:
                prompt_text = f.read()

        # print(f'Input:\n{prompt_text}\n')

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.args.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=self.args.length + len(encoded_prompt[0]),
            temperature=self.args.temperature,
            top_k=self.args.k,
            top_p=self.args.p,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            num_return_sequences=self.args.num_return_sequences,
        )

        if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

        generated_sequences = []
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            text = text[: text.find(self.args.stop_token) if self.args.stop_token else None]
            text = text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]

            if delimiter in text:
                text = text.split(delimiter)[0].rstrip()
            else:
                text = text.split('\n')[0].rstrip()

            generated_sequences.append(text)
            # print(f'[{generated_sequence_idx}]ruGPT:\n{prompt_text.split('\n')[-1] + text}')

        return generated_sequences


    def detoxify(self, text, num_return_sequences=10, k=3, p=0.5, temperature=10.):
        results = []

        # parameters
        self.args.num_return_sequences = num_return_sequences
        self.args.k = k
        self.args.p = p
        self.args.temperature = temperature
        # here text stands for your sentence
        self.args.length = len(text) + 10


        generated_sequences = self.generate_sequences(text + ' >>> ')
        results.append([re.sub('<pad>', '', x) for x in generated_sequences])
        
        return results[0][0][:self.args.length]