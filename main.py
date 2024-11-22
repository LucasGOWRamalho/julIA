from interface.chat import ChatInterface

if __name__ == "__main__":
    chat = ChatInterface(trained_dir="./results")
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit"]:
            break
        response = chat.get_response(user_input)
        print(f"IA: {response}")
