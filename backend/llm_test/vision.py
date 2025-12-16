import ollama

response = ollama.chat(
    model='llama3.2-vision:11b',
    messages=[{
        'role': 'user',
        'content': 'give all the text in the image',
        'images': ['android.jpg']
    }]
)

print(response)