from os import path
import torch
from transformers import AutoTokenizer, AutoConfig
from pathlib import Path
import logging
import Log
import toml
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()

class BaseInference:
    def __init__(self):
        config_file_path = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'config.toml')
        self.inference_config = toml.load(config_file_path)
        self.remote_path = 'Grossmend/rudialogpt3_medium_based_on_gpt2'      
        self.device = torch.device("cpu")
        base_path = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), 'models/rugpt')
        self.local_model_path = f'{base_path}/model'
        self.local_tokenizer_path = f'{base_path}/model'
        self.local_onnx_path = f'{base_path}/model/onnx'
        self.config = AutoConfig.from_pretrained(self.local_model_path)          
        self.chat_model = None
        self.chat_tokenizer = self._check_tokenizer_exists()
 

    def check_pytorch_model_exists(self):
        return True if Path(f'{self.local_model_path}/pytorch_model.bin').exists() and self.chat_tokenizer else False

    def _check_tokenizer_exists(self):
        if Path(f'{self.local_tokenizer_path}/vocab.json'):
            self.chat_tokenizer = AutoTokenizer.from_pretrained(self.local_tokenizer_path) 
        else:
            self.chat_tokenizer = AutoTokenizer.from_pretrained(self.remote_path)
            self.chat_tokenizer.save_pretrained(self.local_tokenizer_path)  
        return self.chat_tokenizer      

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
