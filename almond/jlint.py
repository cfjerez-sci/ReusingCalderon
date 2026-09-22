"""Static checks for the Julia pitfalls that cost a round trip each.

Not a parser: a lexer that strips comments, strings and char literals, then
checks bracket balance, block/end balance, and the two mistakes a Python
habit produces -- implicit string concatenation, and a @printf format that
is not a single string literal.
"""
import re, sys, glob

OPENERS = {"function", "for", "while", "if", "let", "begin", "struct",
           "module", "macro", "quote", "try", "do"}

def lex(src):
    """Return (code, spans) with strings/comments blanked but newlines kept."""
    out = []
    i, n = 0, len(src)
    strings = []          # (start, end) of string literals in the ORIGINAL
    prev_sig = ""         # last significant char, for ' disambiguation
    while i < n:
        c = src[i]
        if c == "#":
            if src.startswith("#=", i):
                d = 1; j = i + 2
                while j < n and d:
                    if src.startswith("#=", j): d += 1; j += 2
                    elif src.startswith("=#", j): d -= 1; j += 2
                    else: j += 1
                out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
                i = j; continue
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i)); i = j; continue
        if c == '"':
            triple = src.startswith('"""', i)
            q = '"""' if triple else '"'
            j = i + len(q)
            while j < n:
                if src[j] == "\\": j += 2; continue
                if src.startswith(q, j): j += len(q); break
                j += 1
            strings.append((i, j))
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j; prev_sig = '"'; continue
        if c == "'":
            if prev_sig.isalnum() or prev_sig in "_)]}.":
                out.append("'"); i += 1; continue          # transpose
            j = i + 1
            while j < n:
                if src[j] == "\\": j += 2; continue
                if src[j] == "'": j += 1; break
                j += 1
            out.append(" " * (j - i)); i = j; prev_sig = "c"; continue
        out.append(c)
        if not c.isspace(): prev_sig = c
        i += 1
    return "".join(out), strings

def check(path):
    src = open(path).read()
    code, strings = lex(src)
    probs = []

    # 1. bracket balance
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for k, ch in enumerate(code):
        if ch in "([{": stack.append((ch, k))
        elif ch in ")]}":
            if not stack or stack[-1][0] != pairs[ch]:
                probs.append(f"line {code[:k].count(chr(10))+1}: unmatched '{ch}'")
                break
            stack.pop()
    for ch, k in stack:
        probs.append(f"line {code[:k].count(chr(10))+1}: unclosed '{ch}'")

    # 2. block / end balance, ignoring `end` used as an index
    depth = 0; opens = 0; ends = 0
    for m in re.finditer(r"[\[\]\(\)]|\b[A-Za-z_][A-Za-z_0-9!]*\b", code):
        t = m.group()
        if t in "([": depth += 1; continue
        if t in ")]": depth -= 1; continue
        ln = code[:m.start()].count("\n") + 1
        if t == "end":
            if depth == 0: ends += 1
            continue
        if t in OPENERS:
            if t == "do" and depth > 0:          # `open(f) do io`
                opens += 1; continue
            if depth > 0: continue
            pre = code[max(0, m.start()-200):m.start()]
            line = pre.split("\n")[-1]
            if t == "if" and re.search(r"\S\s*$", line):   # trailing/ternary if
                continue
            if t == "function" or t in OPENERS:
                opens += 1
    if opens != ends:
        probs.append(f"block/end mismatch: {opens} openers, {ends} ends "
                     f"(heuristic; check by eye)")

    # 3. implicit string concatenation
    for (a1, b1), (a2, b2) in zip(strings, strings[1:]):
        between = src[b1:a2]
        if between.strip() == "":
            probs.append(f"line {src[:a1].count(chr(10))+1}: two string "
                         f"literals with nothing between them -- Julia does "
                         f"not concatenate; use *")

    # 4. soft-scope accumulators: a top-level `for` that assigns to a name
    #    bound above it. Julia makes that a NEW LOCAL, the outer name never
    #    changes, and the loop silently computes nothing. `if` and `begin` do
    #    not open a scope, so only for/while/let/try are at issue.
    lines = code.split("\n")
    OPENS = re.compile(r"^\s*(function|for|while|let|try|struct|module|macro|quote)\b")
    ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9!]*)\s*=[^=]")
    toplevel = set()
    depth = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        t = ln.strip()
        if depth == 0:
            m = ASSIGN.match(ln)
            if m:
                toplevel.add(m.group(1))
            if re.match(r"^\s*for\b", ln):
                d, j, hits = 1, i + 1, []
                while j < len(lines) and d > 0:
                    tj = lines[j].strip()
                    if OPENS.match(lines[j]) or re.search(r"\bdo\b\s*$", tj):
                        d += 1
                    if tj == "end" or tj.startswith("end "):
                        d -= 1
                        if d == 0:
                            break
                    if d >= 1:
                        ma = ASSIGN.match(lines[j])
                        if (ma and ma.group(1) in toplevel
                                and not re.match(r"^\s*(local|global)\b", lines[j])):
                            hits.append((j + 1, ma.group(1)))
                    j += 1
                for lineno, name in hits:
                    probs.append(f"line {lineno}: `{name}` is assigned inside a "
                                 f"top-level for loop that began at line {i+1}, "
                                 f"but is also bound at top level -- Julia makes "
                                 f"this a new local and the outer value never "
                                 f"changes. Move the loop into a function, or "
                                 f"declare `global {name}`.")
                i = j
        if OPENS.match(ln) or re.search(r"\bdo\b\s*$", t):
            depth += 1
        if t == "end" or t.startswith("end "):
            depth = max(0, depth - 1)
        i += 1

    # 5. printf format must be ONE literal
    for m in re.finditer(r"@s?printf\(", src):
        k = m.end()
        while k < len(src) and src[k] in " \n": k += 1
        if src[k] != '"':                        # io argument
            c = src.find(",", k)
            k = c + 1
            while k < len(src) and src[k] in " \n": k += 1
        if src[k] != '"':
            probs.append(f"line {src[:m.start()].count(chr(10))+1}: @printf "
                         f"format is not a string literal")
            continue
        lit = next(s for s in strings if s[0] == k)
        j = lit[1]
        while j < len(src) and src[j] in " \n\t*": j += 1
        if src[j] == '"':
            probs.append(f"line {src[:m.start()].count(chr(10))+1}: @printf "
                         f"format spans several literals (must be one)")
    return probs

bad = 0
for f in sorted(glob.glob("*.jl")):
    p = check(f)
    print(f"{f}: {'OK' if not p else str(len(p)) + ' problem(s)'}")
    for x in p:
        print("   ", x); bad += 1
sys.exit(1 if bad else 0)
