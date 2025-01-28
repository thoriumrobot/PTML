import requests
import sys
import openai
from transformers import GPT2Tokenizer

# Replace this with your OpenAI API key
openai.api_key = 

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Your prompt
PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

instruction = '''Add @Nullable annotations to the following Java code snippet where having them would improve code quality. Enclose code in ```java...```:

'''

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
    response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Use the correct model name here
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                {"type": "text", "text": PROMPT.format(instruction=instruction+chunk)}]},
            ],
            max_tokens=200,  # Adjust the token limit as needed
            temperature=0.7,  # Adjust temperature for randomness
        )
    print("Response:", response.json())  # Add this line to inspect the response
    return response.choices[0].message.content

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

