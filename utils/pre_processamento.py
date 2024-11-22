def limpar_texto(texto):
    # Remove caracteres especiais e espaços extras
    return texto.strip().lower()

from transformers import AutoTokenizer

def carregar_tokenizer(model_name='gpt2'):
    return AutoTokenizer.from_pretrained(model_name)

def preparar_dataset(caminho_dados, tokenizer, max_length=128):
    from datasets import load_dataset
    dataset = load_dataset('text', data_files={'train': caminho_dados})
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    return tokenized_dataset
