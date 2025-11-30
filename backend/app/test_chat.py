import requests

API_URL = "http://127.0.0.1:8000/api/v1/chat"


def chat():
    print("=== MediMind Chat Test ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting MediMind chat.")
            break

        payload = {"message": user_input}
        try:
            response = requests.post(API_URL, json=payload)
            data = response.json()
            print(f"MediMind: {data['answer']} (confidence: {data['confidence']:.2f})\n")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    chat()

