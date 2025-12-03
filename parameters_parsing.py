import re

def strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+%", "%", text)
    return text

def remove_at_words(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"\B@\S+", "", text)

    text = text.rstrip()
    return text


def parse_numeric_params(message: str, context_window: int = 40):
    clean_message = strip_html_tags(message)
    clean_message = clean_message.replace("\r\n", "\n").replace("\r", "\n")

    IGNORE_PATTERNS = [
        r"\b\d{4}-\d{2}-\d{2}\b",          # 2025-12-02
        r"\b\d{2}\.\d{2}\.\d{4}\b",        # 02.12.2025
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",   # 12:55 или 12:55:30
        r"\b\d{4}/\d{2}/\d{2}\b",          # 2025/12/02
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",    # 2-12-2025
    ]

    ignore_spans: list[tuple[int, int]] = []
    for pat in IGNORE_PATTERNS:
        for m in re.finditer(pat, clean_message):
            ignore_spans.append(m.span())

    def is_inside_ignored(start: int, end: int) -> bool:
        for s, e in ignore_spans:
            if start >= s and end <= e:
                return True
        return False

    number_pattern = re.compile(
        r"(?:(?<=\s)|(?<=\())(-?\d+(?:\.\d+)?)(\s*%)?"
        r"|"
        r"^(-?\d+(?:\.\d+)?)(\s*%)?"
    )

    params: list[dict] = []

    for i, match in enumerate(number_pattern.finditer(clean_message)):
        if match.group(1) is not None:
            value_str = match.group(1)
            percent_sign = match.group(2)
            start, end = match.span(1)
        else:
            value_str = match.group(3)
            percent_sign = match.group(4)
            start, end = match.span(3)

        if is_inside_ignored(start, end):
            continue

        try:
            value = float(value_str)
        except ValueError:
            continue

        is_percent = percent_sign is not None and "%" in percent_sign

        # контекст = вся строка, где находится число
        line_start = clean_message.rfind("\n", 0, start)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1  

        line_end = clean_message.find("\n", end)
        if line_end == -1:
            line_end = len(clean_message)

        context = clean_message[line_start:line_end]
        clean_message = remove_at_words(clean_message)
        params.append(
            {
                "index": i,
                "tag": f"param {i + 1}",
                "value": value,
                "is_percent": is_percent,
                "raw": clean_message[start:end],
                "start": start,
                "end": end,
                "context": context.strip(),
            }
        )

    return clean_message, params
