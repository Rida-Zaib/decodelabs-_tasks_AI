responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just a program, but I'm running great! How about you?",
    "what is your name": "I'm ChatBot, a simple rule-based assistant built for Project 1.",
    "what can you do": "I can respond to a few basic greetings and questions. Try asking me how I am!",
    "help": "You can say: hello, how are you, what is your name, what can you do, or bye.",
    "bye": "Goodbye! Have a great day!",
    "exit": "Goodbye! Have a great day!",
    "quit": "Goodbye! Have a great day!",
}

EXIT_COMMANDS = {"bye", "exit", "quit"}


def get_response(user_input):
    return responses.get(user_input, "I do not understand. Type 'help' to see what I can respond to.")


def main():
    print("ChatBot: Hello! Type 'bye', 'exit', or 'quit' to end the chat.")

    while True:
        raw_input_text = input("You: ")
        clean_input = raw_input_text.lower().strip()

        if clean_input in EXIT_COMMANDS:
            print(f"ChatBot: {responses[clean_input]}")
            break

        reply = get_response(clean_input)
        print(f"ChatBot: {reply}")


if __name__ == "__main__":
    main()
