from inference_models.tensorrt_inference import TensorRTInference

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
        if inf_model == 'tensorrt':
            return TensorRTInference()
        else:
            raise ValueError(inf_model)
factory = InfModelFactory()