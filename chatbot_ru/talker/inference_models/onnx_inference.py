import os
import torch
from torch.nn import functional as F
import onnxruntime 
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
from pathlib import Path
import numpy as np
import logging
import Log
from inference_models.gen_wrappers import TopKLogitsWarper, TemperatureLogitsWarper, TopPLogitsWarper, NoRepeatNGramLogitsProcessor
#from timeit import default_timer as timer
from onnxruntime.transformers.quantize_helper import QuantizeHelper
from inference_models.base_inference import BaseInference
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()



class OnnxInference(BaseInference):   
    def check_model_locally(self):
        logger.info("Checking ONNX model and downloading it if necessary")
        if not Path(f'{self.local_onnx_path}/rugpt.onnx').exists():
            if not self.check_pytorch_model_exists() :
                raise FileNotFoundError(f'Cannot find Pytorch model for conversion at: {self.local_model_path}. Setup "pytorch" for inference_model first.')
            logger.info("Converting Pytorch model to ONNX model...")
            self.chat_model = MyGPT2LMHeadModel.from_pretrained(self.local_model_path)
            Gpt2Helper.export_onnx(self.chat_model, self.device, os.path.join(self.local_onnx_path, 'rugpt.onnx'))
            logger.info("Optimizing onnx model...")
            optimized_model = optimizer.optimize_model(os.path.join(self.local_onnx_path, 'rugpt.onnx'), model_type='gpt2', num_heads=self.config.n_head, hidden_size=self.config.n_embd)
            optimized_model.save_model_to_file(os.path.join(self.local_onnx_path, 'rugpt_optimized.onnx'))
        return os.path.join(self.local_onnx_path, 'rugpt_optimized.onnx')

    def start_inference(self, model_path):
        logger.info("Starting ONNX inference session...")            
        self.chat_model = onnxruntime.InferenceSession(model_path) 

    def get_example_inputs(self, text, history):    
        # encode the new user input, add parameters and return a tensor in Pytorch
        new_user_input_ids = self.chat_tokenizer.encode(f"|0|{self.get_length_param(text)}|" + text + self.chat_tokenizer.eos_token +  "|1|1|", return_tensors="pt")
        history_tensors = []
        if history:
                history_tensors= self.chat_tokenizer.encode(history + self.chat_tokenizer.eos_token, return_tensors="pt")          
        # append the new user input tokens to the chat history  
        bot_input_ids = torch.cat([history_tensors, new_user_input_ids], dim=-1) if history else new_user_input_ids

        #input_ids = torch.tensor(bot_input_ids, dtype=torch.int64)
        input_ids = bot_input_ids.long()
        attention_mask = torch.tensor([np.ones(len(bot_input_ids[0]))], dtype=torch.float32)
        position_ids = (attention_mask.long().cumsum(-1) - 1)
        position_ids.masked_fill_(position_ids < 0, 0)

        #Empty Past State for generating first word
        empty_past = []
        batch_size = input_ids.size(0)
   
        past_shape = [2, batch_size, self.config.n_head, 0, self.config.n_embd // self.config.n_head]
        for i in range(self.config.n_layer):
            empty_past.append(torch.empty(past_shape).type(torch.float32).to(self.device))

        
        return input_ids, attention_mask, position_ids, empty_past

    def inference_with_io_binding(self, input_ids, position_ids, attention_mask, past):
        #logger.info(f'past sequence length: {past[0].size(3)}, sequence length: {input_ids.size(1)}')
        output_shapes = Gpt2Helper.get_output_shapes(batch_size=input_ids.size(0),
                                                    past_sequence_length=past[0].size(3),
                                                    sequence_length=input_ids.size(1),
                                                    config=self.config)
        #logger.info(f'output shapes: {output_shapes}')
        output_buffers = Gpt2Helper.get_output_buffers(output_shapes, self.device)
        #logger.info(f'output buffers: {output_buffers}')
        io_binding = Gpt2Helper.prepare_io_binding(self.chat_model, input_ids, position_ids, attention_mask, past,
                                                output_buffers, output_shapes)
        self.chat_model.run_with_iobinding(io_binding)

        outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(self.chat_model, output_buffers, output_shapes,
                                                                return_numpy=False)
        return outputs

    def get_answer_ru(self, text, history=None):        
        input_ids, attention_mask, position_ids, past = self.get_example_inputs(text, history)
        batch_size = input_ids.size(0)
        has_eos = torch.zeros(batch_size, dtype=torch.bool)
        all_token_ids = input_ids.clone()
        updated_input_ids = input_ids.clone()
        #inference_time = float(0)
        for _ in range(self.inference_config['num_tokens_to_produce']):
            #start = timer()
            outputs = self.inference_with_io_binding(updated_input_ids, position_ids, attention_mask, past)
            #end = timer()
            #inference_time += end - start
            next_token_logits = outputs[0][:, -1, :]
            next_token_logits = NoRepeatNGramLogitsProcessor(self.inference_config['no_repeat_ngram_size'])(all_token_ids, next_token_logits)
            next_token_logits = TemperatureLogitsWarper(self.inference_config['temperature'])(next_token_logits)
            next_token_logits = TopKLogitsWarper(self.inference_config['top_k'])(next_token_logits)
            next_token_logits = TopPLogitsWarper(self.inference_config['top_p'])(next_token_logits)
            #next_tokens = torch.argmax(next_token_logits, dim=-1)
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)            
            #logger.info(f'next tokens softmax: {next_tokens}')
            has_eos = has_eos | (next_tokens == self.chat_tokenizer.eos_token_id)
            tokens_to_add = next_tokens.masked_fill(has_eos, self.chat_tokenizer.eos_token_id)
            all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            updated_input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(self.device) 
            #logger.info(f'updated input ids: {updated_input_ids}')
            position_ids = (position_ids[:,-1] + 1).reshape(batch_size,1)
            #logger.info(f'updated position ids: {position_ids}')
            attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(self.device)
            #logger.info(f'updated attention mask: {attention_mask}')    
            past = []
            for i in range(self.config.n_layer):
                past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], np.ndarray) else outputs[i + 1].clone().detach()
                past.append(past_i.to(self.device))
            #logger.info(f'past sequence: {len(past)}')
            if torch.all(has_eos):
                break
        #logger.info(f'Inference time only: {inference_time}')
        return self.chat_tokenizer.decode(all_token_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True), self.chat_tokenizer.decode(all_token_ids[0], skip_special_tokens=True)

class OnnxInferenceFp16(OnnxInference):
    def check_model_locally(self):
        if not Path(f'{self.local_onnx_path}/rugpt_optimized_fp16.onnx').exists():
            OnnxInference.check_model_locally(self)              
            logger.info("Converting onnx model to half-precision model...")
            optimized_model = optimizer.optimize_model(os.path.join(self.local_onnx_path, 'rugpt.onnx'), model_type='gpt2', num_heads=self.config.n_head, hidden_size=self.config.n_embd)
            optimized_model.convert_model_float32_to_float16()
            optimized_model.save_model_to_file(os.path.join(self.local_onnx_path, 'rugpt_optimized_fp16.onnx'))            
        return os.path.join(self.local_onnx_path, 'rugpt_optimized_fp16.onnx')

class OnnxInferenceInt8(OnnxInference):
    def check_model_locally(self):
        if not Path(f'{self.local_onnx_path}/rugpt_optimized_int8.onnx').exists():
            OnnxInference.check_model_locally(self) 
            logger.info("Quantization of optimized ONNX model...") 
            QuantizeHelper.quantize_onnx_model(f'{self.local_onnx_path}/rugpt_optimized.onnx', f'{self.local_onnx_path}/rugpt_optimized_int8.onnx')
        return os.path.join(self.local_onnx_path, 'rugpt_optimized_int8.onnx')
