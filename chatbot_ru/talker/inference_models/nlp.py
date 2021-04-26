from inference_models.pytorch_inference import PytorchInference
from inference_models.onnx_inference import OnnxInference, OnnxInferenceFp16, OnnxInferenceInt8

class NLP:
    def load_model(self, model_type):
        model_path = model_type.check_model_locally()
        return model_type.start_inference(model_path)

class ModelLoader:
    def load_model(self, serializable, inf_model):
        newmodel = factory.get_model(inf_model)
        serializable.load_model(newmodel)
        return newmodel

class InfModelFactory:
    def get_model(self, inf_model):
        if inf_model == 'pytorch':
            return PytorchInference()
        elif inf_model in 'onnx':
            return OnnxInference()
        elif inf_model == 'onnx_fp16':
            return OnnxInferenceFp16()
        elif inf_model == 'onnx_int8':
            return OnnxInferenceInt8()
        else:
            raise ValueError(inf_model)
factory = InfModelFactory()