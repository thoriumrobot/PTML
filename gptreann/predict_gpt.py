import re
import sys
import json
import openai

# Replace this with your OpenAI API key
openai.api_key = 

# Regex patterns to capture fields/methods that can hold annotations:
# Very simplified and will miss complex scenarios, but decent for demonstration.
FIELD_PATTERN = re.compile(r'^(?:public|protected|private)?\s+[\w\<\>\[\]]+\s+\w+\s*(?:=.*)?;')
METHOD_PATTERN = re.compile(r'^(?:public|protected|private)?\s+[\w\<\>\[\]]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{?')

def get_candidate_lines(java_code):
    """
    Returns a dictionary { line_number: (full_line, 'field' or 'method') }
    for all lines that match either field or method patterns.
    """
    lines = java_code.splitlines()
    candidates = {}
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if FIELD_PATTERN.match(stripped):
            candidates[i] = (line, 'field')
        elif METHOD_PATTERN.match(stripped):
            candidates[i] = (line, 'method')
    return candidates

def ask_gpt_for_nullable_decisions(full_code, candidates):
    """
    Sends the entire code plus the list of candidate lines to GPT-4o-mini.
    Expects a JSON response mapping line numbers to "yes" or "no".
    """
    # Format the candidates as a list of "line: content"
    candidate_text = "\n".join(
        f"{line_number}: {text.strip()}"
        for line_number, (text, _) in candidates.items()
    )

    # The user instruction / prompt for GPT-4o-mini:
    prompt = f"""
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses.

Here is the entire Java file for context:
{full_code}

Below are the lines identified as potential places for @Nullable annotations.
Please analyze each line in the context of the entire code. For each line, respond "yes" if @Nullable should be added, or "no" otherwise.

Return your answer as strict JSON in the format:

{{
  "<line_number>": "yes" or "no",
  ...
}}

List only the lines given, with no extra keys.

Candidate lines:
{candidate_text}

IMPORTANT: Return JSON ONLY.
"""

    # Call GPT-4o-mini
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Replace with correct model name or endpoint
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt.strip()}]},
        ],
        max_tokens=500,
        temperature=0.0,  # Set low temperature for more deterministic output
    )

    # We expect raw JSON back; let's parse it.
    content = response.choices[0].message.content
    print("Response:", response.json())  # Add this line to inspect the response
    try:
        decisions = json.loads(content)
    except json.JSONDecodeError:
        # If GPT returns something that isn't strict JSON, handle gracefully.
        # For demo, just return everything as "no"
        decisions = {}
        for ln in candidates.keys():
            decisions[str(ln)] = "no"

    return decisions

def insert_nullable(original_line):
    """
    Insert '@Nullable ' immediately before the type in the line.
    For example:
      "private String name;" -> "private @Nullable String name;"
      "public int getX()"   -> "public @Nullable int getX()"
    This is simplistic and may need refinement for real-world code.
    """
    # Find the first match for typical Java type references after any access modifier
    match = re.search(r'(public|protected|private)?(\s+)([\w\<\>\[\]]+)', original_line)
    if match:
        # Reconstruct the line with @Nullable
        start_idx = match.start(3)  # start of the type
        return original_line[:start_idx] + "@Nullable " + original_line[start_idx:]
    else:
        # No safe place found (should not happen with our patterns), return unchanged
        return original_line

def rewrite_code_with_nullable(full_code, decisions):
    """
    Go through code line by line. If line number is in decisions with "yes", insert @Nullable.
    Return the modified code as a string.
    """
    lines = full_code.splitlines()
    for i, line in enumerate(lines, start=1):
        decision = decisions.get(str(i), "no")
        if decision.lower() == "yes":
            lines[i-1] = insert_nullable(line)
    return "\n".join(lines)

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        java_code = f.read()

    # 1) Identify candidate lines (fields & methods).
    candidates = get_candidate_lines(java_code)

    if not candidates:
        print(f"No suitable fields or methods found in '{input_file}'. Writing original code to '{output_file}'.")
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(java_code)
        return

    # 2) Ask GPT-4o-mini for "yes"/"no" for each candidate line.
    decisions = ask_gpt_for_nullable_decisions(java_code, candidates)

    # 3) Rewrite the code by inserting @Nullable on all lines that got a "yes" from GPT.
    new_code = rewrite_code_with_nullable(java_code, decisions)

    # 4) Output the updated code
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(new_code)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.java> <output_file.java>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
