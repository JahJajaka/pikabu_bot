import torch
from transformers import AutoModelForCausalLM
from inference_models.base_inference import BaseInference
import logging
import Log
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()


class PytorchInference(BaseInference):
    def check_model_locally(self):
        logger.info("Checking PyTorch model and downloading it if necessary")
        return self.local_model_path if self.check_pytorch_model_exists() else self.remote_path

    def start_inference(self, model_path):
        logger.info(f'Prepare local PyTorch model for inference...')
        self.chat_model = AutoModelForCausalLM.from_pretrained(model_path)
        if model_path == self.remote_path: self.chat_model.save_pretrained(self.local_model_path)

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
            no_repeat_ngram_size=self.inference_config['no_repeat_ngram_size'],
            do_sample=True,
            top_k=self.inference_config['top_k'],
            top_p=self.inference_config['top_p'],
            temperature = self.inference_config['temperature'],
            mask_token_id=self.chat_tokenizer.mask_token_id,
            eos_token_id=self.chat_tokenizer.eos_token_id,
            unk_token_id=self.chat_tokenizer.unk_token_id,
            pad_token_id=self.chat_tokenizer.pad_token_id,
            device='cpu',
        )

        # pretty print last ouput tokens from bot
        return self.chat_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True), self.chat_tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
