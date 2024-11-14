from modelo import carregar_modelo
from treinamento import treinar_modelo

def iniciar_treinamento():
    tokenizer, model = carregar_modelo('gpt2')
    treinar_modelo(model, tokenizer, 'dataset/initial_corpus.txt')

if __name__ == "__main__":
    iniciar_treinamento()
