# Rule-Based AI Chatbot — Project 1

**DecodeLabs — Artificial Intelligence Industrial Training Kit (Batch 2026)**

A simple, fully rule-based chatbot built with pure control flow and dictionary
lookups — no AI models, no APIs. Understands **20 languages**, including
Roman Urdu, using Unicode-script detection and keyword matching.

## Files

| File | What it is |
|---|---|
| `chatbot.py` | Base version — English only, the core Project 1 requirement |
| `chatbot_multilingual.py` | Extended version — 20 languages (script detection + keyword matching) |
| `chatbot_frontend.html` | Web chat UI for the multilingual chatbot (open directly in any browser) |

## How to run

```bash
python chatbot_multilingual.py
```

Then just type — for example:

```
You: salam
ChatBot: Walaikum Assalam! Aap kaisay hain?

You: bonjour
ChatBot: Bonjour ! Comment puis-je vous aider aujourd'hui ?

You: bye
ChatBot: Goodbye! Have a great day!
```

> **Windows CMD users:** run `chcp 65001` first so non-English scripts
> (Urdu, Arabic, Chinese, etc.) display correctly. VS Code / PyCharm
> terminals and Mac/Linux terminals don't need this.

For the web UI, just open `chatbot_frontend.html` in any browser — no
install needed.

## How language detection works

1. **Script-based languages** (Arabic, Urdu, Persian, Hindi, Bengali,
   Chinese, Japanese, Korean, Russian) are detected via **Unicode
   character ranges** — fully deterministic, no guessing.
2. **Latin-script languages** (English, Roman Urdu, French, Spanish,
   German, Italian, Portuguese, Turkish, Indonesian, Swahili) are
   detected via **keyword matching** — counting how many of that
   language's signature words appear in the message.

## Key concepts practiced

Control flow, `while` loops, dictionaries, the `.get()` lookup-with-fallback
pattern, string sanitization, and basic Unicode/regex-based rule design.

---
*Submitted as part of the DecodeLabs AI Internship — Project 1.*
