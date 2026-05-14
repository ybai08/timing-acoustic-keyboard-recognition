# Prompt Lists

These prompt files are used by the collection app.

Use synthetic prompts only. Do not type real passwords, private messages, account names, addresses, or anything you would not want stored in a local research dataset.

Each non-empty line is treated as one prompt.

`random_strings.txt` contains the original short random strings plus the common casual English phrase prompts. The English phrases intentionally use lowercase letters and spaces only, which keeps that dataset focused on common conversation without adding punctuation classes too early.

`letter_mixes.txt` contains lowercase letter-only strings designed for individual key acoustics. These prompts are intentionally not normal words: they mix nearby and far-apart keys so the dataset includes many different letter transitions while still keeping the character set simple.
