import re
import sys
import json
import openai

# Replace this with your OpenAI API key
openai.api_key = 

############################################
# REGEX PATTERNS
############################################

FIELD_PATTERN = re.compile(
    r'^(?:public|protected|private)?'
    r'(?:\s+static|\s+final|\s+volatile|\s+transient)*'
    r'\s+[\w\<\>\[\]]+\s+\w+\s*(?:=.*)?;'
)

# Note the updated pattern: we're using standard numeric capturing groups
# rather than `(?<prefix>...)`.
METHOD_PATTERN = re.compile(
    r'^(?:public|protected|private)?'                       # Access modifier
    r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'  # Possibly other modifiers
    r'\s+([\w\<\>\[\]]+)'                                   # (Group 1) Return type
    r'\s+(\w+)'                                             # (Group 2) Method name
    r'\s*\(([^)]*)\)'                                       # (Group 3) Parameter list
    r'(?:\s*throws\s+[\w\.\,]+)?'                           # possibly throws
    r'\s*\{?'                                               # optional starting {
)

def get_candidate_items(java_code):
    lines = java_code.splitlines()
    candidates = {}

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        if FIELD_PATTERN.match(stripped):
            key = f"line:{i}/type:field"
            candidates[key] = {
                "category": "field",
                "line_number": i,
                "content": line,
                "full_line": line,
                "param_index": None
            }
            continue

        method_match = METHOD_PATTERN.match(stripped)
        if method_match:
            return_type = method_match.group(1)
            method_name = method_match.group(2)
            params_str = method_match.group(3)

            # Candidate for method return
            return_key = f"line:{i}/type:return"
            candidates[return_key] = {
                "category": "return",
                "line_number": i,
                "content": f"{return_type} {method_name}(...)", 
                "full_line": line,
                "param_index": None
            }

            # Candidate for method params
            params_str = params_str.strip()
            if params_str:
                param_list = [p.strip() for p in params_str.split(",")]
                for idx, param in enumerate(param_list, start=1):
                    param_key = f"line:{i}/type:param/index:{idx}"
                    candidates[param_key] = {
                        "category": "param",
                        "line_number": i,
                        "content": param,
                        "full_line": line,
                        "param_index": idx
                    }

    return candidates

def ask_gpt_for_nullable_decisions(full_code, candidates):
    candidate_text = "\n".join(
        f"{key} => {info['content'].strip()}"
        for key, info in candidates.items()
    )

    prompt = f"""
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses.

Below is the entire Java file for context:
{full_code}


We have identified these candidate items (fields, method returns, or method parameters) that might benefit from an @Nullable annotation. Each candidate is labeled with a unique key. For each key, please reply "yes" if it should be @Nullable, or "no" otherwise.

Return your answer strictly as JSON in the format:

{{
  "<candidate_key>": "yes" or "no",
  ...
}}

Where <candidate_key> is exactly one of the keys listed below. DO NOT return extra keys.

Candidates:
{candidate_text}

IMPORTANT: Return JSON ONLY.
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # or the correct model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt.strip()}]}
        ],
        max_tokens=800,
        temperature=0.0  
    )

    content = response.choices[0].message.content
    print("Response:", response.json())  # Add this line to inspect the response
    try:
        decisions = json.loads(content)
    except json.JSONDecodeError:
        decisions = {}
        for k in candidates.keys():
            decisions[k] = "no"

    return decisions

def insert_nullable_in_field_or_return(line):
    pattern = re.compile(r'^(.*?)([\w\<\>\[\]]+)(\s+\w.*)$')
    match = pattern.match(line.strip())
    if match:
        prefix = match.group(1)
        typ    = match.group(2)
        suffix = match.group(3)
        return f"{prefix}@Nullable {typ}{suffix}"
    return line

def insert_nullable_in_param(full_signature, param_index):
    """
    Insert '@Nullable ' in the nth parameter of the given method signature line.
    """

    # Numeric capturing groups version:
    method_pattern = re.compile(
        r'^((?:public|protected|private)?'
        r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'
        r'\s+[\w\<\>\[\]]+\s+\w+\s*\()'  # group(1): everything up to '('
        r'([^)]*)'                      # group(2): parameters
        r'(\).*)$'                      # group(3): closing parenthesis and maybe '{'
    )

    m = method_pattern.match(full_signature.strip())
    if not m:
        return full_signature

    prefix = m.group(1)     # up to '('
    params_str = m.group(2) # inside (...)
    suffix = m.group(3)     # after ')'

    split_params = [p.strip() for p in params_str.split(',') if p.strip()]

    # If param_index is out of range, do nothing
    if param_index < 1 or param_index > len(split_params):
        return full_signature

    # Update the target param
    target_param = split_params[param_index - 1]
    # Insert @Nullable
    param_pattern = re.compile(r'^(?:final\s+)?([\w\<\>\[\]]+)(\s+\w+)$')
    pm = param_pattern.match(target_param)
    if pm:
        if target_param.startswith("final "):
            new_param = "final @Nullable " + pm.group(1) + pm.group(2)
        else:
            new_param = "@Nullable " + pm.group(1) + pm.group(2)
    else:
        new_param = "@Nullable " + target_param

    split_params[param_index - 1] = new_param
    new_params_str = ", ".join(split_params)
    return prefix + new_params_str + suffix

def rewrite_code_with_nullable(full_code, decisions):
    lines = full_code.splitlines()
    from collections import defaultdict
    grouped = defaultdict(list)

    # Group decisions by line
    for key, val in decisions.items():
        if val.lower() != "yes":
            continue

        line_match = re.search(r'line:(\d+)', key)
        cat_match  = re.search(r'/type:(\w+)', key)
        idx_match  = re.search(r'/index:(\d+)', key)

        if not line_match or not cat_match:
            continue

        line_no = int(line_match.group(1))
        category = cat_match.group(1)
        param_idx = int(idx_match.group(1)) if idx_match else None

        grouped[line_no].append((category, param_idx))

    for i in range(len(lines)):
        line_no = i + 1
        if line_no not in grouped:
            continue

        line_actions = grouped[line_no]
        categories = [c[0] for c in line_actions]

        # If "field" or "return" is among them, insert @Nullable at the type
        if "field" in categories or "return" in categories:
            lines[i] = insert_nullable_in_field_or_return(lines[i])

        # Then handle parameters in ascending order
        param_actions = [(cat, idx) for (cat, idx) in line_actions if cat == "param"]
        if param_actions:
            param_actions.sort(key=lambda x: x[1])
            new_line = lines[i]
            for (cat, pidx) in param_actions:
                new_line = insert_nullable_in_param(new_line, pidx)
            lines[i] = new_line

    return "\n".join(lines)

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        java_code = f.read()

    candidates = get_candidate_items(java_code)
    if not candidates:
        print(f"No suitable items found in '{input_file}'. Writing original code to '{output_file}'.")
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(java_code)
        return

    decisions = ask_gpt_for_nullable_decisions(java_code, candidates)
    new_code = rewrite_code_with_nullable(java_code, decisions)

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(new_code)

    print(f"Finished. Updated file written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.java> <output_file.java>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
