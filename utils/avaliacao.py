from transformers import pipeline
import os

def calcular_perplexidade(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss)

def salvar_interacoes(novas_interacoes, caminho='dataset/novas_interacoes.txt'):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, 'a', encoding='utf-8') as f:
        for interacao in novas_interacoes:
            f.write(f"{interacao}\n")
