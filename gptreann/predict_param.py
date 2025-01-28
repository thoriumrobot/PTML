import re
import sys
import json
import openai

# Replace this with your OpenAI API key
openai.api_key = 

############################################
# REGEX PATTERNS
############################################

# Very naive patterns for demonstration only.

# Matches fields like:
#   private String name;
#   public List<String> names = new ArrayList<>();
FIELD_PATTERN = re.compile(
    r'^(?:public|protected|private)?'
    r'(?:\s+static|\s+final|\s+volatile|\s+transient)*'
    r'\s+[\w\<\>\[\]]+\s+\w+\s*(?:=.*)?;'
)

# Matches a method signature line with return type, name, and parameter list:
#   public String getName(String prefix, int count) throws SomeException {
# (Group 1: possible modifiers, Group 2: return type, Group 3: method name, Group 4: parameters)
METHOD_PATTERN = re.compile(
    r'^(?:public|protected|private)?'            # Access modifier
    r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'  # Possibly other modifiers
    r'\s+([\w\<\>\[\]]+)'                        # (Group 1) Return type
    r'\s+(\w+)'                                  # (Group 2) Method name
    r'\s*\(([^)]*)\)'                            # (Group 3) Parameter list
    r'(?:\s*throws\s+[\w\.\,]+)?'                # possibly throws
    r'\s*\{?'                                    # optional starting {
)

############################################
# CANDIDATE EXTRACTION
############################################

def get_candidate_items(java_code):
    """
    Parse the Java code line by line. Identify fields, method return types, and method parameters.
    Return a dictionary keyed by:
       "line:<line_number>/type:<field|return|param>/index:<param_index?>"
    Values are a dict with:
       {
         "category": "field" | "return" | "param",
         "line_number": int,
         "content": <the relevant snippet to annotate>,
         "full_line": <the entire line>
         "param_index": <index of parameter if category=="param"> or None
       }
    """
    lines = java_code.splitlines()
    candidates = {}

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # ----------------------------
        # Check for field
        # ----------------------------
        if FIELD_PATTERN.match(stripped):
            # We'll treat the entire type portion as the "content" for annotation
            key = f"line:{i}/type:field"
            candidates[key] = {
                "category": "field",
                "line_number": i,
                "content": line,  # The full field line for context
                "full_line": line,
                "param_index": None
            }
            continue

        # ----------------------------
        # Check for method signature
        # ----------------------------
        method_match = METHOD_PATTERN.match(stripped)
        if method_match:
            return_type = method_match.group(1)  # e.g. "String"
            method_name = method_match.group(2)  # e.g. "getName"
            params_str = method_match.group(3)   # e.g. "String s, int x"

            # (a) The method return type as a candidate
            return_key = f"line:{i}/type:return"
            candidates[return_key] = {
                "category": "return",
                "line_number": i,
                "content": f"{return_type} {method_name}(...)",  # Short snippet
                "full_line": line,
                "param_index": None
            }

            # (b) Each parameter as a candidate
            # Split by commas. This is naive as it won't handle generics with commas in <...>.
            # For demonstration we do a simple split:
            params_str = params_str.strip()
            if params_str:
                param_list = [p.strip() for p in params_str.split(",")]
                for idx, param in enumerate(param_list, start=1):
                    # param might look like: "String s" or "final List<String> items"
                    # We'll store it as is and rely on rewriting carefully
                    param_key = f"line:{i}/type:param/index:{idx}"
                    candidates[param_key] = {
                        "category": "param",
                        "line_number": i,
                        "content": param,    # e.g. "String s"
                        "full_line": line,   # the entire method signature line
                        "param_index": idx
                    }

    return candidates

############################################
# GPT CALL
############################################

def ask_gpt_for_nullable_decisions(full_code, candidates):
    """
    Sends the entire code plus a list of candidate items (fields, return, params).
    Expects a JSON response mapping each candidate key -> "yes" or "no".
    """
    # Build a textual list of candidate lines
    # Example: 
    #   line:12/type:field -> private String name;
    #   line:15/type:return -> public String getName(...) 
    #   line:15/type:param/index:1 -> String prefix
    candidate_text = "\n".join(
        f"{key} => {info['content'].strip()}"
        for key, info in candidates.items()
    )

    # Prompt for GPT
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
        temperature=0.0  # more deterministic
    )

    content = response.choices[0].message.content
    print("Response:", response.json())  # Add this line to inspect the response
    try:
        decisions = json.loads(content)
    except json.JSONDecodeError:
        # Fallback if GPT doesn't return valid JSON
        decisions = {}
        for k in candidates.keys():
            decisions[k] = "no"

    return decisions

############################################
# CODE REWRITING
############################################

def insert_nullable_in_field_or_return(line):
    """
    Insert '@Nullable ' immediately before the type in a field or return-type declaration.
    E.g., "private String name;" -> "private @Nullable String name;"
          "public String getName() {" -> "public @Nullable String getName() {"
    """
    # This is very naive. We'll look for something that looks like:
    #   optional_modifiers + type + at least one space + name
    # We skip the method name for the "return" since it's presumably after the type
    # We'll capture the first group after modifiers that looks like a type token.

    pattern = re.compile(r'^(.*?)([\w\<\>\[\]]+)(\s+\w.*)$')
    match = pattern.match(line.strip())
    if match:
        prefix = match.group(1)  # e.g. "private "
        typ    = match.group(2)  # e.g. "String"
        suffix = match.group(3)  # e.g. " name;" or " getName() {"
        return f"{prefix}@Nullable {typ}{suffix}"
    return line  # fallback if we cannot parse

def insert_nullable_in_param(full_signature, param_index):
    """
    Insert '@Nullable ' in the nth parameter of the given method signature line.
    
    Example line:
       "public void myMethod(String s, int x) {"
    param_index = 1 => "public void myMethod(@Nullable String s, int x) {"

    Returns the modified line.
    """

    # We first capture: modifiers + return_type + method_name + "(" + param_list + ")" + possible remainder
    method_pattern = re.compile(
        r'^(?<prefix>(?:public|protected|private)?'
        r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'
        r'\s+[\w\<\>\[\]]+\s+\w+\s*\()'
        r'(?<params>[^)]*)'
        r'(?<suffix>\).*)$', 
        re.VERBOSE
    )
    # Python's `re` doesn't support inline group names this way, so let's do capturing groups manually:
    method_pattern = re.compile(
        r'^((?:public|protected|private)?'
        r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'
        r'\s+[\w\<\>\[\]]+\s+\w+\s*\()'  # group 1 => everything up to '('
        r'([^)]*)'                      # group 2 => parameters
        r'(\).*)$'                      # group 3 => closing parenthesis plus remainder
    )

    m = method_pattern.match(full_signature.strip())
    if not m:
        return full_signature  # fallback

    prefix = m.group(1)  # includes '(' at the end
    params_str = m.group(2)
    suffix = m.group(3)

    # Split parameters. This naive split won't handle generics with commas inside <...>,
    # or annotation usage, etc. But it's sufficient for a demo.
    split_params = [p.strip() for p in params_str.split(',') if p.strip()]

    # Safety check: if param_index is out of range, do nothing
    if param_index < 1 or param_index > len(split_params):
        return full_signature

    # Insert "@Nullable " in front of that param's type
    # We'll do a small pattern to insert after possible modifiers (like "final")
    target_param = split_params[param_index - 1]
    # e.g., "String s", or "final List<String> items"

    # Let's see if we can place @Nullable
    # For example, if target_param is "final List<String> items"
    # Insert after "final " => "final @Nullable List<String> items"
    # or if "String s", => "@Nullable String s"

    param_pattern = re.compile(r'^(?:final\s+)?([\w\<\>\[\]]+)(\s+\w+)$')
    pm = param_pattern.match(target_param)
    if pm:
        # The param has optional "final" then the type, then the variable name
        # We'll insert @Nullable after "final " if it exists, else just at start
        if target_param.startswith("final "):
            new_param = "final @Nullable " + pm.group(1) + pm.group(2)
        else:
            new_param = "@Nullable " + pm.group(1) + pm.group(2)
    else:
        # fallback - just slap @Nullable in front
        new_param = "@Nullable " + target_param

    split_params[param_index - 1] = new_param
    new_params_str = ", ".join(split_params)

    return prefix + new_params_str + suffix

def rewrite_code_with_nullable(full_code, decisions):
    """
    Go line by line; if we find a candidate with "yes", insert @Nullable.
    We handle:
      - fields
      - method return types
      - each parameter
    """
    lines = full_code.splitlines()

    # For convenience, group decisions by line_number
    #   grouped[line_number] = list of (category, param_index or None)
    from collections import defaultdict
    grouped = defaultdict(list)
    for key, val in decisions.items():
        if val.lower() != "yes":
            continue
        # key looks like: "line:12/type:param/index:1"
        # parse line number
        line_match = re.search(r'line:(\d+)', key)
        category_match = re.search(r'/type:(\w+)', key)
        param_index_match = re.search(r'/index:(\d+)', key)

        if not line_match or not category_match:
            continue

        line_no = int(line_match.group(1))
        category = category_match.group(1)
        if param_index_match:
            param_idx = int(param_index_match.group(1))
        else:
            param_idx = None

        grouped[line_no].append((category, param_idx))

    # Now we apply them line by line
    for i in range(len(lines)):
        line_no = i + 1
        if line_no not in grouped:
            continue

        # If we have multiple instructions for the same line, handle them
        line_actions = grouped[line_no]

        # The easiest approach is to rewrite from the bottom up:
        #  1) if there's a "return" or "field" => insert @Nullable in that line
        #  2) if there are parameters => insert @Nullable into each parameter in ascending param_index
        # Because rewriting multiple times can be tricky, we'll do in a stable manner:
        # First handle field/return (which do not rely on param_index),
        # then handle param indexes in ascending order.

        # Check if "field" or "return" is among them
        categories = [c[0] for c in line_actions]
        if "field" in categories or "return" in categories:
            lines[i] = insert_nullable_in_field_or_return(lines[i])

        # Now handle parameters in ascending order of param_index
        param_actions = [(cat, idx) for (cat, idx) in line_actions if cat == "param"]
        if param_actions:
            # sort by param_index ascending
            param_actions.sort(key=lambda x: x[1])
            # re-insert for each param
            new_line = lines[i]
            for (cat, pidx) in param_actions:
                new_line = insert_nullable_in_param(new_line, pidx)
            lines[i] = new_line

    return "\n".join(lines)


############################################
# MAIN SCRIPT
############################################

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        java_code = f.read()

    # 1) Identify fields, method returns, and method parameters
    candidates = get_candidate_items(java_code)
    if not candidates:
        print(f"No suitable items found in '{input_file}'. Writing original code to '{output_file}'.")
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(java_code)
        return

    # 2) Ask GPT for "yes"/"no" for each candidate
    decisions = ask_gpt_for_nullable_decisions(java_code, candidates)

    # 3) Rewrite the code
    new_code = rewrite_code_with_nullable(java_code, decisions)

    # 4) Output
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(new_code)

    print(f"Finished. Updated file written to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.java> <output_file.java>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
