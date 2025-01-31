import re
import sys
import json
import openai
from collections import defaultdict

# Replace this with your OpenAI API key
openai.api_key = ""

############################################
# HELPER PARSING FUNCTIONS
############################################

def parse_generic(s, i):
    """
    Parse a generic block starting at index i (where s[i] == '<').
    Returns the generic block (with no spaces inside angle brackets) and the new index.
    """
    start = i
    depth = 0
    n = len(s)
    while i < n:
        if s[i] == '<':
            depth += 1
        elif s[i] == '>':
            depth -= 1
            if depth == 0:
                i += 1  # include the closing '>'
                break
        i += 1
    block = s[start:i]
    block = remove_spaces_in_angle_brackets(block)
    return block, i

def tokenize_preserving_generics(s):
    """
    Tokenize a string by whitespace, but merge tokens that are part of a type’s
    generic block or array brackets. For example:
    
      "final List < String >[] data"
      
    will become:
      
      ["final", "List<String>[]", "data"]
    
    This function peeks ahead so that if whitespace is immediately followed by
    a '<' or '[' (which we want to join to the preceding token) we do not flush.
    """
    tokens = []
    token = ""
    i = 0
    n = len(s)
    while i < n:
        if s[i].isspace():
            j = i
            while j < n and s[j].isspace():
                j += 1
            # If the next non-space char is '<' or '[', do not flush the current token.
            if j < n and (s[j] == '<' or s[j] == '['):
                i = j
                continue
            else:
                if token:
                    tokens.append(token)
                    token = ""
                i = j
                continue
        elif s[i] == '<':
            generic_block, new_i = parse_generic(s, i)
            token += generic_block
            i = new_i
        else:
            token += s[i]
            i += 1
    if token:
        tokens.append(token)
    return tokens

def remove_spaces_in_angle_brackets(s):
    """
    Remove all spaces between < and > pairs (including nested ones).
    For example:
      "List < String >" => "List<String>"
      "Map < String , List < Integer >>" => "Map<String,List<Integer>>"
    """
    out = []
    depth = 0
    for ch in s:
        if ch == '<':
            depth += 1
            out.append(ch)
        elif ch == '>':
            out.append(ch)
            if depth > 0:
                depth -= 1
        else:
            if not (ch.isspace() and depth > 0):
                out.append(ch)
    return "".join(out)

def compress_type(t):
    """
    Remove extraneous spaces around array brackets in a type string.
    E.g., "List<String> []" -> "List<String>[]"
    """
    return re.sub(r'\s*\[\s*\]', '[]', t)

def join_type_tokens(tokens):
    """
    Join tokens that represent a type. If a token starts with '<' or '[',
    append it directly without a space.
    """
    if not tokens:
        return ""
    result = tokens[0]
    for token in tokens[1:]:
        if token.startswith('<') or token.startswith('['):
            result += token
        else:
            result += " " + token
    return result

def parse_single_param(param_str):
    """
    Given a single parameter string, return a tuple:
      (prefix_modifiers, actual_type, variable_name)
      
    Examples:
       "final List<String> data" 
         => ("final", "List<String>", "data")
       "Map < String , List <Integer>>[] arr" 
         => ("", "Map<String,List<Integer>>[]", "arr")
    """
    tokens = tokenize_preserving_generics(param_str)
    if not tokens:
        # fallback
        return ("", param_str, "")
    
    var_name = tokens[-1]
    type_modifiers = tokens[:-1]
    
    # Identify known modifiers and annotations.
    known_mods = {"final", "public", "protected", "private", "static", "abstract",
                  "synchronized", "native", "transient", "volatile"}
    prefix_list = []
    i = 0
    while i < len(type_modifiers):
        t = type_modifiers[i]
        if t in known_mods or t.startswith("@"):
            prefix_list.append(t)
            i += 1
        else:
            break
    prefix = " ".join(prefix_list)
    # Join the remaining tokens so that e.g. "List" and "<String>" are merged without an extra space.
    actual_type = join_type_tokens(type_modifiers[i:])
    actual_type = compress_type(actual_type)
    
    return (prefix, actual_type, var_name)

############################################
# REGEX PATTERNS
############################################

# Field pattern:
# Matches lines such as:
#   private List<String> names;
#   protected Map<String, List<Integer>>[] cache = ...;
#   public static final MyClass<Object> something;
# Group 1 captures the type (e.g., "List<String>") and Group 2 the variable name.
FIELD_PATTERN = re.compile(
    r'^(?:public|protected|private)?'
    r'(?:\s+static|\s+final|\s+volatile|\s+transient|\s+synchronized)*'
    r'\s+([\w\<\>\[\]]+(?:\.[\w\<\>\[\]]+)*?)'
    r'\s+(\w+)\s*(?:=.*)?;'
)

# Method pattern:
# Matches method signatures such as:
#   public List<String> doStuff(Map<String, Integer> param) throws FooException {
#   private MyClass[] process(final List<String> data, int x) {
# Group 1: Return type, Group 2: Method name, Group 3: Parameter list.
METHOD_PATTERN = re.compile(
    r'^(?:public|protected|private)?'
    r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'
    r'\s+([\w\<\>\[\]]+(?:\.[\w\<\>\[\]]+)*)'
    r'\s+(\w+)'
    r'\s*\(([^)]*)\)'
    r'(?:\s*throws\s+[\w\.\,]+)?'
    r'\s*\{?'
)

############################################
# CANDIDATE EXTRACTION
############################################

def split_method_params(params_str):
    """
    Split a method parameter string by top-level commas, ignoring commas inside
    angle-bracket pairs (e.g., generics like <String, Integer>).
    
    Example:
       "List<String> x, Map<String, List<Integer>> y"
    => ["List<String> x", "Map<String, List<Integer>> y"]
    """
    param_list = []
    buffer = []
    depth = 0  # Track nesting depth for '<' and '>'
    i = 0
    while i < len(params_str):
        ch = params_str[i]
        if ch == '<':
            depth += 1
            buffer.append(ch)
        elif ch == '>':
            buffer.append(ch)
            if depth > 0:
                depth -= 1
        elif ch == ',' and depth == 0:
            param_list.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(ch)
        i += 1
    if buffer:
        param_list.append("".join(buffer).strip())
    return param_list

def get_candidate_items(java_code):
    """
    Identify candidate items (fields, method return types, method parameters)
    from the Java code. Each candidate is assigned a unique key.
    
    Example candidate keys:
        "line:10/type:field"
        "line:12/type:return"
        "line:12/type:param/index:2"
    """
    lines = java_code.splitlines()
    candidates = {}
    
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Try matching a field.
        field_match = FIELD_PATTERN.match(stripped)
        if field_match:
            key = f"line:{i}/type:field"
            candidates[key] = {
                "category": "field",
                "line_number": i,
                "full_line": line,
            }
            continue
        
        # Try matching a method.
        method_match = METHOD_PATTERN.match(stripped)
        if method_match:
            # Record the method return type.
            return_key = f"line:{i}/type:return"
            candidates[return_key] = {
                "category": "return",
                "line_number": i,
                "full_line": line,
            }
            # Process each parameter.
            params_str = method_match.group(3).strip()
            if params_str:
                param_list = split_method_params(params_str)
                for idx, param in enumerate(param_list, start=1):
                    param_key = f"line:{i}/type:param/index:{idx}"
                    candidates[param_key] = {
                        "category": "param",
                        "line_number": i,
                        "full_line": line,
                        "param_index": idx,
                        "param_str": param
                    }
    return candidates

############################################
# GPT CALL
############################################

def ask_gpt_for_nullable_decisions(full_code, candidates):
    """
    Sends the full code plus a list of candidate items to GPT, expecting a JSON
    response with keys and values "yes" or "no". (If the JSON cannot be decoded,
    the default decision for every candidate is "no".)
    """
    candidate_text = []
    for key, info in candidates.items():
        if info["category"] == "param":
            candidate_text.append(f"{key} => (parameter) {info['param_str']}")
        elif info["category"] == "field":
            candidate_text.append(f"{key} => (field) {info['full_line'].strip()}")
        elif info["category"] == "return":
            candidate_text.append(f"{key} => (method return) {info['full_line'].strip()}")
    candidate_text = "\n".join(candidate_text)
    
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
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.strip()}
        ],
        max_tokens=800,
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    try:
        decisions = json.loads(content)
    except json.JSONDecodeError:
        decisions = {}
        for k in candidates.keys():
            decisions[k] = "no"
    
    return decisions

############################################
# CODE REWRITING
############################################

def parse_field_or_return(line):
    """
    Given a field declaration or a method signature line, split it into:
      - prefix (modifiers before the type)
      - type token (which may be built from several tokens, e.g. "List" and "<String>")
      - suffix (everything after the type)
    
    For example:
      "private List < String > name = new ArrayList<>();"
    might be split into:
      prefix: "private"
      type: "List<String>"
      suffix: "name = new ArrayList<>();"
    """
    tokens = tokenize_preserving_generics(line)
    known_mods = {"public", "protected", "private", "static", "final",
                  "abstract", "synchronized", "native", "volatile", "transient"}
    prefix_tokens = []
    idx_of_type = -1
    for i, t in enumerate(tokens):
        if t in known_mods or t.startswith("@"):
            prefix_tokens.append(t)
        else:
            idx_of_type = i
            break
    if idx_of_type == -1:
        return line, "", ""
    
    # Merge tokens that belong to the type (e.g. "List" and "<String>")
    type_tokens = []
    type_tokens.append(tokens[idx_of_type])
    j = idx_of_type + 1
    while j < len(tokens) and (tokens[j].startswith('<') or tokens[j].startswith('[')):
        type_tokens.append(tokens[j])
        j += 1
    type_token = join_type_tokens(type_tokens)
    type_token = compress_type(type_token)
    suffix_tokens = tokens[j:]
    prefix = " ".join(prefix_tokens)
    suffix = " ".join(suffix_tokens)
    
    return prefix, type_token, suffix

def insert_nullable_in_field_or_return(line):
    """
    Insert '@Nullable' after the modifiers but before the type token.
    For example:
      "private List<String> name;" becomes
      "private @Nullable List<String> name;"
    """
    prefix, typ, suffix = parse_field_or_return(line)
    if not typ:
        return line  # fallback
    prefix = prefix.strip()
    if prefix:
        return f"{prefix} @Nullable {typ} {suffix}"
    else:
        return f"@Nullable {typ} {suffix}"

def insert_nullable_in_param(full_signature, param_index):
    """
    Insert '@Nullable' in the nth parameter of a method signature.
    For example, given:
      "public void myMethod(String s, List<String> data) {"
    and param_index=2, the result will be:
      "public void myMethod(String s, @Nullable List<String> data) {"
    """
    pattern = re.compile(
        r'^(\s*(?:public|protected|private)?'
        r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'
        r'\s+[\w\<\>\[\]\.]+\s+\w+\s*\()'
        r'([^)]*)'
        r'(\).*)$'
    )
    m = pattern.match(full_signature.strip())
    if not m:
        return full_signature
    
    prefix  = m.group(1)
    params  = m.group(2)
    suffix  = m.group(3)
    
    param_list = split_method_params(params)
    if not (1 <= param_index <= len(param_list)):
        return full_signature
    
    old_param = param_list[param_index - 1]
    prefix_mods, actual_type, var_name = parse_single_param(old_param)
    if prefix_mods.strip():
        new_param = prefix_mods + " @Nullable " + actual_type + " " + var_name
    else:
        new_param = "@Nullable " + actual_type + " " + var_name
    param_list[param_index - 1] = new_param
    new_params = ", ".join(param_list)
    return prefix + new_params + suffix

def rewrite_code_with_nullable(full_code, decisions):
    """
    Rewrite the Java code by inserting @Nullable annotations wherever the
    decisions (provided as a dict mapping candidate keys to "yes"/"no")
    indicate it should be added.
    """
    lines = full_code.splitlines()
    grouped = defaultdict(list)
    
    # Group decisions by line number.
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
    
    # Process each line that has modifications.
    for i in range(len(lines)):
        line_no = i + 1
        if line_no not in grouped:
            continue
        
        actions = grouped[line_no]
        # For field or method return type candidates:
        if any(cat == "field" for cat, _ in actions) or any(cat == "return" for cat, _ in actions):
            lines[i] = insert_nullable_in_field_or_return(lines[i])
        
        # For parameter candidates (apply in ascending order of parameter index).
        param_actions = [(cat, idx) for (cat, idx) in actions if cat == "param"]
        if param_actions:
            param_actions.sort(key=lambda x: x[1])
            new_line = lines[i]
            for _, pidx in param_actions:
                new_line = insert_nullable_in_param(new_line, pidx)
            lines[i] = new_line
    
    return "\n".join(lines)

############################################
# MAIN
############################################

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        java_code = f.read()
    
    # Step 1: Identify candidate items.
    candidates = get_candidate_items(java_code)
    if not candidates:
        print(f"No suitable items found in '{input_file}'. Writing original code to '{output_file}'.")
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(java_code)
        return
    
    # Step 2: Ask GPT for decisions ("yes" or "no") for each candidate.
    decisions = ask_gpt_for_nullable_decisions(java_code, candidates)
    
    # Step 3: Rewrite the code based on those decisions.
    new_code = rewrite_code_with_nullable(java_code, decisions)
    
    # Step 4: Write the updated code.
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(new_code)
    
    print(f"Finished. Updated file written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.java> <output_file.java>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

