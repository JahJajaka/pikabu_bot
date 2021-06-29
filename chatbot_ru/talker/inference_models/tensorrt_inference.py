import logging
import Log
from timeit import default_timer as timer
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = Log.get_logger()


class TensorRTInference:
    def check_model_locally(self):
        logger.info("Checking PyTorch model and downloading it if necessary")
        return self.local_model_path if self.check_pytorch_model_exists() else self.remote_path

    def start_inference(self, model_path):
        logger.info(f'Prepare local TensorRT model for inference...')


    def get_answer_ru(self, batch_text, batch_history):
        text_batch = []
        for text, history in zip(batch_text,batch_history):
            comb_string = f"|0|{self.get_length_param(text)}|" + text + self.chat_tokenizer.eos_token +  "|1|1|"
            if history:
                comb_string = history + self.chat_tokenizer.eos_token + comb_string
            text_batch.append(comb_string)
        logger.info(f'Model input text batch size: {len(text_batch)}')
        encodings_dict = self.chat_tokenizer.batch_encode_plus(text_batch, padding=True)
        input_ids_shape = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64).shape[-1]
        input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64).to(self.device)
        #logger.info(f'Pytorch input ids: {input_ids}')
        start = timer()
        chat_history_ids = self.chat_model.generate(
            input_ids,
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
            device=self.device,
        ).detach().clone()
        torch.cuda.empty_cache()
        end = timer()
        logger.info(f'Pytorch inference time only: {end - start}')
        answer = []
        new_history = []
        for i, output in enumerate(chat_history_ids):
            answer.append(self.chat_tokenizer.decode(output[input_ids_shape:], skip_special_tokens=True))
            new_history.append(self.chat_tokenizer.decode(output, skip_special_tokens=True))

        return answer , new_history
