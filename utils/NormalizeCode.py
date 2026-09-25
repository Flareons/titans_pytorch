import re

def normalize_code(code: str) -> str:
    """
    Normalize C/C++ source code for vulnerability detection (Big-Vul).

    Steps:
    1. Remove comments safely
    2. Normalize whitespace
    3. Normalize long string literals
    4. Normalize large numeric literals

    Keep:
    - function names
    - variable names
    - API names
    - security-related strings
    - macros
    """

    result = []
    i = 0
    n = len(code)

    in_string = False
    in_char = False

    # =========================
    # 1. Remove comments
    # =========================

    while i < n:
        c = code[i]

        # string literal
        if c == '"' and not in_char:
            in_string = not in_string
            result.append(c)
            i += 1
            continue

        # char literal
        if c == "'" and not in_string:
            in_char = not in_char
            result.append(c)
            i += 1
            continue


        if not in_string and not in_char:

            # single line comment //
            if c == '/' and i + 1 < n and code[i+1] == '/':
                i += 2

                while i < n and code[i] != '\n':
                    i += 1

                continue


            # multi line comment /*
            if c == '/' and i + 1 < n and code[i+1] == '*':
                i += 2

                while i + 1 < n:
                    if code[i] == '*' and code[i+1] == '/':
                        i += 2
                        break
                    i += 1

                continue


        result.append(c)
        i += 1


    code = ''.join(result)


    # =========================
    # 2. Normalize whitespace
    # =========================

    code = "\n".join(
        line.strip()
        for line in code.splitlines()
        if line.strip()
    )


    # =========================
    # 3. Normalize long strings
    # Keep security/API strings
    # =========================

    def replace_string(match):
        value = match.group(0)

        # remove quotes
        content = value[1:-1]

        # giữ string ngắn
        # ví dụ:
        # "schemaFlagsEx"
        # "searchFlags"
        if len(content) <= 20:
            return value

        return '"STRING"'


    code = re.sub(
        r'"([^"\\]|\\.)*"',
        replace_string,
        code
    )


    # =========================
    # 4. Normalize large numbers
    # Keep:
    # 0,1,2,10,100
    # =========================

    code = re.sub(
        r'\b\d{4,}\b',
        'NUM',
        code
    )


    return code