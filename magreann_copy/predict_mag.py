import requests
import sys
from transformers import AutoTokenizer

# Your prompt
MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

instruction = '''Add @Nullable annotations to the following Java code snippet where having them would improve code quality. Enclose code in ```java...```:

'''

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("ise-uiuc/Magicoder-S-DS-6.7B")

def chunk_text(text, max_length):
    tokens = tokenizer.tokenize(text)
    current_chunk = []
    for token in tokens:
        current_chunk.append(token)
        if (token.endswith(';') or token == '}') and len(current_chunk) >= max_length:
            yield tokenizer.convert_tokens_to_string(current_chunk)
            current_chunk = []
    if current_chunk:
        yield tokenizer.convert_tokens_to_string(current_chunk)

def process_chunk(chunk):
    url = "http://127.0.0.1:8080/api/predict/"
    data = {
        "data": [MAGICODER_PROMPT.format(instruction=instruction+chunk), 1, 2048]
    }
    response = requests.post(url, json=data)
    print("Response:", response.json())  # Add this line to inspect the response
    return response.json().get('data', [""])[0]  # Safely access 'data' key

def main(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    processed_content = ""
    for chunk in chunk_text(content, 192):  # Reduced chunk size
        processed_chunk = process_chunk(chunk)
        # Extract the code from the processed chunk
        start = processed_chunk.find('```java')
        end = processed_chunk.find('```', start + 1)
        if start != -1 and end != -1:
            processed_content += processed_chunk[start + len('```java'):end].strip()
        else:
            processed_content += chunk

    with open(output_file, 'w') as file:
        file.write(processed_content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.java output_file.java")
    else:
        main(sys.argv[1], sys.argv[2])

