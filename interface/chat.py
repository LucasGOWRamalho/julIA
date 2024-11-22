from ia.model import Pipeline

class ChatInterface:
    def __init__(self, model_name="gpt2", trained_dir="./results"):
        self.pipeline = Pipeline(model_name=model_name)
        self.pipeline.model = self.pipeline.model.from_pretrained(trained_dir)
        self.pipeline.tokenizer = self.pipeline.tokenizer.from_pretrained(trained_dir)

    def get_response(self, user_input):
        return self.pipeline.generate_response(user_input)
