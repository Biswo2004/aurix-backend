"""Input checks applied before a message is sent to the model."""

import re

MAX_MESSAGE_CHARS = 8000

_BLOCKED_PHRASES = (
    "child porn",
    "child pornography",
    "child sexual",
    "sexualize a child",
    "sexualise a child",
    "csam",
    "how to make a bomb",
    "build a bomb",
    "make a pipe bomb",
    "how to make sarin",
    "how to make ricin",
    "how to make meth",
    "synthesize methamphetamine",
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def check_message(message: str) -> tuple[bool, str]:
    """Return ``(allowed, refusal_text)``.

    Length and a small set of blocked phrases are the whole policy. A refusal
    is written so it can be streamed back to the user.
    """
    if message is None or not message.strip():
        return False, "Please enter a message so I can help."
    if len(message) > MAX_MESSAGE_CHARS:
        return (
            False,
            "That message is too long. Please shorten it to under 8,000 characters and try again.",
        )
    if _CONTROL.search(message):
        return False, "That message contains characters I can't process. Please rephrase it."
    lowered = message.lower()
    for phrase in _BLOCKED_PHRASES:
        if phrase in lowered:
            return (
                False,
                "I can't help with that request. If you have a question about your files or another topic, please rephrase it.",
            )
    return True, ""
