from transformers import AutoTokenizer, AutoModelForCausalLM

def carregar_modelo(model_name='gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
