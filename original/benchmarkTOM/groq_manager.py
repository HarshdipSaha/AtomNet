
from groq import Groq

GROQ_KEYS = [
"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",
"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",
"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",
"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",
"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",
"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",
"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"
]
key_index = 0

def get_client():
    global key_index

    key = GROQ_KEYS[key_index]

    key_index = (key_index + 1) % len(GROQ_KEYS)

    return Groq(api_key=key)
