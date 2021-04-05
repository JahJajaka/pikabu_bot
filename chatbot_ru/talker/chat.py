import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

class NLP:
    def __init__(self):
        remote_path = 'Grossmend/rudialogpt3_medium_based_on_gpt2'
        base_path = '/talker/models/rugpt'
        local_model_path = f'{base_path}/model'
        local_tokenizer_path = f'{base_path}/model'
        if Path(f'{local_model_path}/pytorch_model.bin').exists() and Path(f'{local_tokenizer_path}/vocab.json').exists() :
            self.chat_model = AutoModelForCausalLM.from_pretrained(local_model_path)
            self.chat_tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
        else:
            self.chat_model = AutoModelForCausalLM.from_pretrained(remote_path)
            self.chat_model.save_pretrained(local_model_path)
            self.chat_tokenizer = AutoTokenizer.from_pretrained(remote_path)
            self.chat_tokenizer.save_pretrained(local_tokenizer_path)

    def get_length_param(self, text: str) -> str:
        tokens_count = len(self.chat_tokenizer.encode(text))
        if tokens_count <= 15:
            len_param = '1'
        elif tokens_count <= 50:
            len_param = '2'
        elif tokens_count <= 256:
            len_param = '3'
        else:
            len_param = '-'
        return len_param

    def get_answer_ru(self, text, history):        
        # encode the new user input, add parameters and return a tensor in Pytorch
        new_user_input_ids = self.chat_tokenizer.encode(f"|0|{self.get_length_param(text)}|" + text + self.chat_tokenizer.eos_token +  "|1|1|", return_tensors="pt")
        history_tensors = []
        if history:
                history_tensors= self.chat_tokenizer.encode(history + self.chat_tokenizer.eos_token, return_tensors="pt")          
        # append the new user input tokens to the chat history  
        bot_input_ids = torch.cat([history_tensors, new_user_input_ids], dim=-1) if history else new_user_input_ids
        # generated a response
        chat_history_ids = self.chat_model.generate(
            bot_input_ids,
            num_return_sequences=1,
            max_length=512,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature = 0.6,
            mask_token_id=self.chat_tokenizer.mask_token_id,
            eos_token_id=self.chat_tokenizer.eos_token_id,
            unk_token_id=self.chat_tokenizer.unk_token_id,
            pad_token_id=self.chat_tokenizer.pad_token_id,
            device='cpu',
        )

        # pretty print last ouput tokens from bot
        return self.chat_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True), self.chat_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)