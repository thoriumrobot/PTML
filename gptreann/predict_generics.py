import re
import sys
import json
import openai

# Replace this with your OpenAI API key
openai.api_key = 

############################################
# REGEX PATTERNS
############################################

# 1) FIELD_PATTERN
#    Example matches:
#       private List<String> names;
#       protected Map<String, List<Integer>>[] cache = ...
#       public static final MyClass<Object> something;
#    Group 1 captures the entire "type" portion (e.g. "List<String>", "Map<String, List<Integer>>[]")
#    Group 2 captures the variable name (e.g. "names", "cache", "something")
FIELD_PATTERN = re.compile(
    r'^(?:public|protected|private)?'
    r'(?:\s+static|\s+final|\s+volatile|\s+transient|\s+synchronized)*'
    r'\s+([\w\<\>\[\]]+(?:\.[\w\<\>\[\]]+)*?)'  # Group 1 => Type portion (with generics)
    r'\s+(\w+)\s*(?:=.*)?;'                    # Group 2 => Variable name
)

# 2) METHOD_PATTERN
#    Example matches:
#       public List<String> doStuff(Map<String, Integer> param) throws FooException {
#       private MyClass[] process(final List<String> data, int x) {
#    Group 1 => Return type (possibly generic, array)
#    Group 2 => Method name
#    Group 3 => Parameter list (all text inside parentheses)
METHOD_PATTERN = re.compile(
    r'^(?:public|protected|private)?'
    r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'
    r'\s+([\w\<\>\[\]]+(?:\.[\w\<\>\[\]]+)*)'  # (Group 1) Return type with generics
    r'\s+(\w+)'                               # (Group 2) Method name
    r'\s*\(([^)]*)\)'                         # (Group 3) Parameter list (raw text)
    r'(?:\s*throws\s+[\w\.\,]+)?'
    r'\s*\{?'  # optional brace
)

############################################
# CANDIDATE EXTRACTION
############################################

def get_candidate_items(java_code):
    """
    Identify:
      - Fields
      - Method return types
      - Method parameters
    Return a dict: { candidate_key: candidate_info }
    
    candidate_key e.g. "line:10/type:field"
                     "line:12/type:return"
                     "line:12/type:param/index:2"
    """
    lines = java_code.splitlines()
    candidates = {}

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        # Try field pattern
        field_match = FIELD_PATTERN.match(stripped)
        if field_match:
            # entire type + var name
            key = f"line:{i}/type:field"
            candidates[key] = {
                "category": "field",
                "line_number": i,
                "full_line": line,
            }
            continue

        # Try method pattern
        method_match = METHOD_PATTERN.match(stripped)
        if method_match:
            return_type = method_match.group(1)
            method_name = method_match.group(2)
            params_str  = method_match.group(3)

            # (a) Method return type
            return_key = f"line:{i}/type:return"
            candidates[return_key] = {
                "category": "return",
                "line_number": i,
                "full_line": line,
            }

            # (b) Each parameter
            params_str = params_str.strip()
            if params_str:
                # Split by commas (naive; breaks if generics had commas, but typical usage is fine).
                param_list = [p.strip() for p in params_str.split(",")]
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
    Sends the entire code plus a list of candidate items, 
    expects JSON with candidate_key: "yes"/"no".
    """
    # Collect textual list
    candidate_text = []
    for key, info in candidates.items():
        if info["category"] == "param":
            # param: show the param text
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
        # If GPT doesn't return valid JSON, default to "no" for all
        decisions = {}
        for k in candidates.keys():
            decisions[k] = "no"

    return decisions

############################################
# CODE REWRITING
############################################

def parse_field_or_return(line):
    """
    For a field line like:
      "private List<String> name = new ArrayList<>();"
    or a method signature line like:
      "public Map<String, SomeClass[]> doSomething( ... ) {"

    We want to separate the portion before the type, 
    the type itself (including generics and array brackets),
    and the portion after.

    We'll do a naive token-based approach to find the first legitimate "type" token.
    """
    # We define possible leading tokens as: public, protected, private, static, final, etc.
    # Then the next chunk that looks like a type (with generics) is the "type".
    # Then everything else is suffix.

    # We'll use a quick approach: 
    # 1. Split on whitespace. 
    # 2. Move through tokens capturing them as "modifiers" until we find a token that 
    #    looks like a type. We must handle generics in one token, e.g. "List<String>".
    #    Usually code doesn't put spaces inside generics.
    # 3. Put the remainder in "suffix".

    # This approach won't handle type tokens with embedded spaces like "Map <String, List<Integer>>", 
    # but typical code doesn't do that.
    tokens = line.strip().split()
    if len(tokens) < 2:
        return (line, "", "")  # fallback

    # Common Java keywords that can appear before type
    known_mods = {
        "public", "protected", "private",
        "static", "final", "abstract", 
        "synchronized", "native", "volatile", "transient"
    }

    # Start collecting prefixes, find type token
    prefix_tokens = []
    type_token = None
    idx_of_type = -1

    for i, t in enumerate(tokens):
        # Once we find something that doesn't look like a known modifier
        # we'll guess it's our type token (including generics).
        if t in known_mods:
            prefix_tokens.append(t)
        else:
            type_token = t
            idx_of_type = i
            break

    if type_token is None:
        return (line, "", "")  # fallback if we didn't find a type

    # Everything after that belongs to "suffix"
    suffix_tokens = tokens[idx_of_type + 1:]

    prefix = " ".join(prefix_tokens)
    suffix = " ".join(suffix_tokens)

    # Rebuild the line as: prefix + " " + type_token + " " + suffix
    # We'll handle spacing carefully below
    return (prefix, type_token, suffix)

def insert_nullable_in_field_or_return(line):
    """
    Insert '@Nullable' after all modifiers (if any), but before the type token.
    E.g.:
      "private List<String> name;" -> "private @Nullable List<String> name;"
      "public static final Map<String, Foo> doSomething() {" -> "public static final @Nullable Map<String, Foo> doSomething() {"
    """
    prefix, typ, suffix = parse_field_or_return(line)
    if not typ:
        return line  # fallback

    # Reconstruct: prefix + @Nullable + type + suffix
    # Handle spacing carefully
    if prefix.strip():
        new_line = prefix.strip() + " @Nullable " + typ + " " + suffix
    else:
        new_line = "@Nullable " + typ + " " + suffix

    return new_line

def parse_single_param(param_str):
    """
    Given a single parameter string, e.g.:
      "final List<String> data"
      "Map<String, Foo>[][] arr"
      "int x"
    We want (prefixMods, typeToken, varName).
    
    We'll do the same naive logic: split on whitespace, the last token is presumably varName,
    the rest are modifiers + type. We insert @Nullable after 'final' (if present) but before the actual type.

    This approach won't handle inline annotations or multi-word types with spaces in generics.
    Typical Java code doesn't do that, so this is good enough for a demonstration.
    """
    # Split by whitespace
    tokens = param_str.strip().split()
    if not tokens:
        return ("", param_str)  # fallback

    # The final token is presumably variable name, e.g. "data", "arr", "x"
    var_name = tokens[-1]
    type_tokens = tokens[:-1]

    # Check if the first token is 'final', we might preserve that
    # or if there are multiple modifiers, e.g. final static isn't typical in params, but let's not break it.
    # We'll gather all leading modifiers that appear in known_mods or start with '@'
    known_mods = {"final"}
    i = 0
    mod_tokens = []
    while i < len(type_tokens):
        t = type_tokens[i]
        # If it's an annotation or recognized mod, treat as prefix
        if t.lower() in known_mods or t.startswith("@"):
            mod_tokens.append(t)
            i += 1
        else:
            break
    # Now, everything else from i onward is the type with generics
    actual_type = " ".join(type_tokens[i:])  # Usually a single token, e.g. "List<String>", but let's allow spaces if any.

    # Reassemble
    prefix = " ".join(mod_tokens)
    return (prefix, actual_type, var_name)

def insert_nullable_in_param(full_signature, param_index):
    """
    Insert '@Nullable' in the nth parameter of the method signature.
    E.g.:
      "public void myMethod(String s, List<String> data) {"
      param_index=2 => "public void myMethod(String s, @Nullable List<String> data) {"
    """
    # 1) Split method signature around parentheses using a capturing group approach
    pattern = re.compile(
        r'^(\s*(?:public|protected|private)?'
        r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'
        r'\s+[\w\<\>\[\]\.]+\s+\w+\s*\()'  # group(1) => everything up to '('
        r'([^)]*)'                        # group(2) => parameters
        r'(\).*)$'                        # group(3) => everything after ')'
    )
    m = pattern.match(full_signature.strip())
    if not m:
        # If it doesn't match, fallback
        return full_signature

    prefix  = m.group(1)  # up to '(' (includes '(' at the end)
    params  = m.group(2)  # inside the parentheses
    suffix  = m.group(3)  # includes ')' and possibly "throws ..." or "{" etc.

    # 2) Split params by commas
    param_list = [p.strip() for p in params.split(",")]
    # if param_index is out of range, do nothing
    if not (1 <= param_index <= len(param_list)):
        return full_signature

    # 3) Insert @Nullable in the targeted param
    old_param = param_list[param_index - 1]
    (prefix_mods, actual_type, var_name) = parse_single_param(old_param)

    # Rebuild the param with @Nullable
    # If there's any prefix mods like "final", keep them first
    if prefix_mods.strip():
        new_param = prefix_mods + " @Nullable " + actual_type + " " + var_name
    else:
        new_param = "@Nullable " + actual_type + " " + var_name

    param_list[param_index - 1] = new_param
    new_params = ", ".join(param_list)

    # 4) Reconstruct
    return prefix + new_params + suffix

def rewrite_code_with_nullable(full_code, decisions):
    """
    Goes line by line. If decisions says "yes" for a given candidate, 
    insert @Nullable in that line (for fields/return) or in the parameter (for param).
    """
    lines = full_code.splitlines()

    from collections import defaultdict
    grouped = defaultdict(list)
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
        actions = grouped[line_no]
        # Possibly multiple actions on same line (e.g. return + param).
        # We'll do field/return first, then param (in ascending param index).
        categories = [a[0] for a in actions]
        if "field" in categories or "return" in categories:
            lines[i] = insert_nullable_in_field_or_return(lines[i])

        # Now handle each param
        param_actions = [(cat, idx) for (cat, idx) in actions if cat == "param"]
        if param_actions:
            # Sort by ascending param index
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

    # Step 1: Identify fields, returns, params
    candidates = get_candidate_items(java_code)
    if not candidates:
        print(f"No suitable items found in '{input_file}'. Writing original code to '{output_file}'.")
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(java_code)
        return

    # Step 2: Ask GPT for "yes"/"no" for each candidate
    decisions = ask_gpt_for_nullable_decisions(java_code, candidates)

    # Step 3: Rewrite the code
    new_code = rewrite_code_with_nullable(java_code, decisions)

    # Step 4: Write
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(new_code)

    print(f"Finished. Updated file written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.java> <output_file.java>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
