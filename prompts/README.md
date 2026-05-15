# Prompt Lists

These prompt files are used by the collection app.

Use synthetic prompts only. Do not type real passwords, private messages, account names, addresses, or anything you would not want stored in a local research dataset.

Each non-empty line is treated as one prompt.

`english_phrases.txt` contains common casual English phrase prompts. These prompts intentionally use lowercase letters and spaces only, which keeps that dataset focused on common conversation without adding punctuation classes too early.

`random_strings.txt` contains the original short random strings plus lowercase letter-only mixes designed for individual key acoustics. The letter-mix prompts are intentionally not normal words: they mix nearby and far-apart keys so the dataset includes many different letter transitions while still keeping the character set simple.

`underused_key_words.txt` contains targeted English word prompts based on the current acoustic model's class counts and confusion matrix. Use it when you want more examples of rare or missing letters such as `z`, `x`, `q`, `j`, `v`, `f`, `b`, `p`, `g`, `c`, and `k`, while still typing normal word-like prompts instead of random strings.

`single_letters.txt` contains repeated single-letter prompts such as `aaaaaa`, `bbbbbb`, and `cccccc`. Use it for clean isolated-key collection when you want several examples of the same key in one short trial.
