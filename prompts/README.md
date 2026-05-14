# Prompt Lists

These prompt files are used by the collection app.

Use synthetic prompts only. Do not type real passwords, private messages, account names, addresses, or anything you would not want stored in a local research dataset.

Each non-empty line is treated as one prompt.

`english_phrases.txt` contains common casual English phrase prompts. These prompts intentionally use lowercase letters and spaces only, which keeps that dataset focused on common conversation without adding punctuation classes too early.

`random_strings.txt` contains the original short random strings plus lowercase letter-only mixes designed for individual key acoustics. The letter-mix prompts are intentionally not normal words: they mix nearby and far-apart keys so the dataset includes many different letter transitions while still keeping the character set simple.
