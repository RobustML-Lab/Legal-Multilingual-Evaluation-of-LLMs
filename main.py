import ollama

stream = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    stream=True
)

for message in stream:
    print(message['message']['content'], end='', flush=True)