import datetime
import re
import typing

SEPARATORS = "-':.!,\r\n\t\f\v "

REGEX_SEPARATORS = r"-':\.!,\s"

DATETIME_PATTERN = re.compile(
    "".join(
        (
            r"(20[012]\d{1})",
            f"[{REGEX_SEPARATORS}]?",
            r"(\d{2})",
            f"[{REGEX_SEPARATORS}]?",
            r"(\d{2})",
            "T?",
            r"(\d{2})",
            f"[{REGEX_SEPARATORS}]?",
            r"(\d{2})",
            f"[{REGEX_SEPARATORS}]?",
            r"(\d{2})",
            r"(Z|[+-][01]\d:\d{2}|[+-][01]\d\d{2}|[+-][01]\d)?",
        )
    )
)

DATE_PATTERN = re.compile(
    "".join(
        (
            r"(20[012]\d{1})",
            f"[{REGEX_SEPARATORS}]?",
            r"(\d{2})",
            f"[{REGEX_SEPARATORS}]?",
            r"(\d{2})",
        )
    )
)


def camel_to_snake(string: str) -> str:
    result = ""
    lower = False
    many_upper = False
    replace_characters = {".", "-", "(", ")"}
    for index, character in enumerate(string):
        if character.isupper():
            if lower:
                result = f"{result}_{character.lower()}"
            else:
                result = f"{result}{character.lower()}"
                many_upper = index > 0
            lower = False
        elif character.isspace() or character in replace_characters:
            result = f"{result}_"
            lower = False
            many_upper = False
        elif character == "_":
            result = f"{result}_"
            lower = False
            many_upper = False
        else:
            if many_upper:
                result = f"{result[:-1]}_{result[-1]}{character}"
            else:
                result = f"{result}{character}"
            lower = True
            many_upper = False
    return result


def stem_and_date(
    stem: str, trim_prefixes: list[str], trim_suffixes: list[str]
) -> tuple[str, typing.Optional[str]]:
    stem = stem.strip()
    for prefix in trim_prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    for suffix in trim_suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    date = None
    match = DATETIME_PATTERN.search(stem)
    if match is None:
        match = DATE_PATTERN.search(stem)
        if match is not None:
            date = datetime.datetime(
                int(match[1]),
                int(match[2]),
                int(match[3]),
                tzinfo=datetime.timezone.utc,
            )
    else:
        if match[7] is None or match[7] == "Z":
            date = datetime.datetime(
                int(match[1]),
                int(match[2]),
                int(match[3]),
                int(match[4]),
                int(match[5]),
                int(match[6]),
                tzinfo=datetime.timezone.utc,
            )
        else:
            hours = int(match[7][0:3])
            if len(match[7]) > 3:
                minutes = int(match[7][0] + match[7][-2:])
            else:
                minutes = 0
            date = datetime.datetime(
                int(match[1]),
                int(match[2]),
                int(match[3]),
                int(match[4]),
                int(match[5]),
                int(match[6]),
                tzinfo=datetime.timezone(
                    datetime.timedelta(hours=hours, minutes=minutes)
                ),
            )
    if match is not None:
        stem = stem[0 : match.start()] + stem[match.end() :]
    stem = camel_to_snake(stem.strip(SEPARATORS))
    if date is not None:
        date = date.isoformat().replace("+00:00", "Z")
    return stem, date
