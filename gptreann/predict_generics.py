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

def split_method_params(params_str):
    """
    Split a method parameter string by top-level commas, ignoring any commas
    that appear inside angle-bracket pairs (e.g. generics like <String, Integer>).
    
    Example:
       "List<String> x, Map<String, List<Integer>> y"
    => ["List<String> x", "Map<String, List<Integer>> y"]
    """
    param_list = []
    buffer = []
    depth = 0  # Track nesting depth of '<' ... '>'
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
            # Top-level comma => separates parameters
            param_list.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(ch)
        i += 1

    # Add the last piece
    if buffer:
        param_list.append("".join(buffer).strip())

    return param_list


def tokenize_preserving_generics(s):
    """
    Tokenize a string by whitespace, but merge consecutive tokens if we're inside <...> depth.
    Also merges them if they come consecutively while that depth is > 0.
    
    This way "List < String >" => ["List<String>"] rather than ["List", "<", "String", ">"].
    Example:
       "final List < String >[] data"
    =>  ["final", "List<String>[]", "data"]
    """
    parts = s.strip().split()
    tokens = []
    buffer = []
    depth = 0

    for p in parts:
        # Add p to buffer
        # Update depth by counting < and >
        # We'll do it character by character so multiple < > in the same token are handled.
        had_lt_gt = False
        for ch in p:
            if ch == '<':
                depth += 1
                had_lt_gt = True
            elif ch == '>':
                had_lt_gt = True
                if depth > 0:
                    depth -= 1

        buffer.append(p)

        # If depth is zero after adding this piece, 
        # we finalize the buffer as one token
        if depth == 0:
            merged = " ".join(buffer)
            # If we had < or > anywhere, let's remove internal spaces
            # E.g. "List < String >[]" => "List<String>[]"
            if had_lt_gt:
                merged = remove_spaces_in_angle_brackets(merged)
            tokens.append(merged)
            buffer = []

    # leftover
    if buffer:
        merged = " ".join(buffer)
        merged = remove_spaces_in_angle_brackets(merged)
        tokens.append(merged)

    return tokens


def remove_spaces_in_angle_brackets(s):
    """
    Remove all spaces between < and > pairs. Also handle nested generics.
    E.g. "List < String >" => "List<String>"
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


def parse_single_param(param_str):
    """
    Given a single parameter string, return (prefix_mods, actual_type, var_name).
    This handles possible 'final' or annotation tokens at the start,
    spaces in generics, etc.
    
    For example:
       "final List<String> data"
         => ("final", "List<String>", "data")
       "Map < String , List <Integer>>[] arr"
         => ("", "Map<String,List<Integer>>[]", "arr")
    """
    tokens = tokenize_preserving_generics(param_str)
    if not tokens:
        # fallback
        return ("", param_str, "")

    # The last token should be the variable name (possibly with no brackets).
    var_name = tokens[-1]
    type_modifiers = tokens[:-1]

    # Some tokens at the front could be "final", or annotations like "@NonNull"
    # We'll treat any token that looks like a known modifier or starts with '@' as part of prefix
    known_mods = {"final", "public", "protected", "private", "static", "abstract",
                  "synchronized", "native", "transient", "volatile"}
    prefix_list = []
    i = 0
    while i < len(type_modifiers):
        t = type_modifiers[i]
        # If it's an annotation or recognized mod, treat as prefix
        if t in known_mods or t.startswith("@"):
            prefix_list.append(t)
            i += 1
        else:
            break

    prefix = " ".join(prefix_list)
    actual_type = " ".join(type_modifiers[i:])

    return (prefix, actual_type, var_name)


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
    
    candidate_key e.g.:
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

        # Try field pattern
        field_match = FIELD_PATTERN.match(stripped)
        if field_match:
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
                # Use our custom top-level comma split, so generics with commas won't break
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

    # Example OpenAI call (pseudo)
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or the correct model name
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

    We want to separate the portion before the type, the type itself,
    and the portion after. We do a mini-tokenization similar to how we do params.
    """
    # We'll do a whitespace-based tokenization but preserve generics.
    tokens = tokenize_preserving_generics(line)

    # Common Java keywords that can appear before the type
    known_mods = {"public","protected","private",
                  "static","final","abstract",
                  "synchronized","native","volatile","transient"}
    prefix_tokens = []
    idx_of_type = -1

    # Find the first token that isn't a known modifier/annotation
    for i, t in enumerate(tokens):
        # strip punctuation like '=' or '{' if it occurs
        # But let's just do a simple check: if t is in known_mods or starts with '@', it's prefix.
        if t in known_mods or t.startswith("@"):
            prefix_tokens.append(t)
        else:
            idx_of_type = i
            break

    if idx_of_type == -1:
        # fallback
        return line, "", ""

    type_token = tokens[idx_of_type]
    suffix_tokens = tokens[idx_of_type+1:]

    prefix = " ".join(prefix_tokens)
    suffix = " ".join(suffix_tokens)

    return prefix, type_token, suffix


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

    prefix = prefix.strip()
    if prefix:
        return f"{prefix} @Nullable {typ} {suffix}"
    else:
        return f"@Nullable {typ} {suffix}"


def insert_nullable_in_param(full_signature, param_index):
    """
    Insert '@Nullable' in the nth parameter of the method signature.
    E.g.:
      "public void myMethod(String s, List<String> data) {"
      param_index=2 => "public void myMethod(String s, @Nullable List<String> data) {"
    """
    # 1) Separate into: prefix (up to '('), params inside '()', suffix
    pattern = re.compile(
        r'^(\s*(?:public|protected|private)?'
        r'(?:\s+static|\s+final|\s+abstract|\s+synchronized|\s+native)*'
        r'\s+[\w\<\>\[\]\.]+\s+\w+\s*\()'  # group(1) => everything up to '('
        r'([^)]*)'                        # group(2) => parameters inside parentheses
        r'(\).*)$'                        # group(3) => everything after ')'
    )
    m = pattern.match(full_signature.strip())
    if not m:
        # fallback
        return full_signature

    prefix  = m.group(1)  # up to '(' (includes '(' at end)
    params  = m.group(2)  # inside the parentheses
    suffix  = m.group(3)  # includes ')' and beyond

    # 2) Split params by top-level commas
    param_list = split_method_params(params)

    # If param_index is out of range, do nothing
    if not (1 <= param_index <= len(param_list)):
        return full_signature

    old_param = param_list[param_index - 1]
    (prefix_mods, actual_type, var_name) = parse_single_param(old_param)

    # Rebuild the param with @Nullable
    if prefix_mods.strip():
        new_param = prefix_mods + " @Nullable " + actual_type + " " + var_name
    else:
        new_param = "@Nullable " + actual_type + " " + var_name

    param_list[param_index - 1] = new_param
    new_params = ", ".join(param_list)

    return prefix + new_params + suffix


def rewrite_code_with_nullable(full_code, decisions):
    """
    Goes line by line. If decisions says "yes" for a given candidate, 
    insert @Nullable in that line (for fields/return) or in the parameter (for param).
    """
    lines = full_code.splitlines()
    grouped = defaultdict(list)

    # Group decisions by line number
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

    # Rewrite lines as needed
    for i in range(len(lines)):
        line_no = i + 1
        if line_no not in grouped:
            continue

        actions = grouped[line_no]
        # Possibly multiple actions on the same line (though rare). 
        # We'll do field/return insertion first, then param (in ascending order).
        if any(cat == "field" for cat, _ in actions) or any(cat == "return" for cat, _ in actions):
            lines[i] = insert_nullable_in_field_or_return(lines[i])

        param_actions = [(cat, idx) for (cat, idx) in actions if cat == "param"]
        if param_actions:
            # sort param actions by ascending index
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

    # Step 2: Ask GPT for "yes"/"no" for each candidate (or mock your own logic)
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

