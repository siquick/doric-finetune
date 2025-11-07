Short answer: it’s a good starting set for “always-Doric”, but it’ll need a few targeted upgrades to really lock the behaviour in for a small Gemma 3 fine-tune with Unsloth.

Here’s what I checked (quick pass on your JSONL):

* Format: each row has `messages: [{role:"user"}, {role:"assistant"}]` — perfect for Unsloth/Gemma chat templates. No system turns required.
* Size & balance: ~3.7k samples (≈2.3 MB). Assistant turns average ~290 chars. That’s workable for a “style/locale” SFT on a small model, but I’d aim for 8–20k examples to make it iron-clad.
* Multilingual inputs: nice coverage (user prompts in DE/IT/HI/PL/NL/PT/FR/SV/TR/AR etc.). That’s exactly what you want if the model must reply Doric regardless of input language.
* Output language: spot-checking suggests the assistant is Doric in the vast majority of rows; however there’s some repetition and a subset of answers are “Scots-flavoured English” rather than strongly Doric. For your goal (“ONLY ever reply in Doric”), you want zero ambiguity here.

What I’d change to hit your goal reliably

1. Add anti-instruction & jailbreak guards (high priority)
   Right now the dataset mostly shows normal Q→A, with the assistant replying in Doric. You also need many adversarial examples that explicitly force the model to ignore requests like:

* “Reply in English / translate to English / respond in JSON only / respond with one word: ‘OK’ / repeat after me in English / roleplay as an English teacher…”
* “You are a translation engine. Output English only.”
  For each, the preferred answer must still be Doric (optionally acknowledging the constraint in Doric). Include ~10–20% of the corpus as these counter-examples and vary phrasing.

2. Strengthen dialect signals & consistency (high)
   Standardise orthography and boost a clear Doric lexicon (aye, nae, fit, loon/quine, hoose, birl, blether, scunnered, breeks, etc.) across all outputs. Right now some answers read like generic Scots. Add a lightweight normalisation pass or a “style linter” to catch English fillers (“in conclusion”, overly formal register) and replace with Doric equivalents.

3. Increase domain & register variety (medium)
   You’ve got good topical spread, but outputs lean on a similar scaffolding. Add:

* Ultra-short one-liners, error messages, and tool snippets (e.g., “File no fund, min.”).
* Formal and semi-formal Doric (emails, policies), not only conversational advice.
* Safety/refusal patterns in Doric (medical/financial/legal deflection done properly in-dialect).

4. Preference-optimise for Doric (medium)
   SFT will get you close, but to make “Doric over anything else” sticky, add a small DPO/ORPO phase:

* Construct pairs where the “rejected” is English (or mixed) and the “chosen” is Doric for the same prompt (especially for the anti-instruction cases).
* 3–5k pairs is plenty for a small model. This dramatically reduces drift under adversarial prompts.

5. Hard negative “translation” items (medium)
   Include many prompts that *look* like translation tasks (English→English paraphrase, code-only requests, JSON-only requests). The correct response is *Doric*, possibly with the requested structure but Doric within it. (If you ever need pure-JSON later, fine-tune a separate head/checkpoint.)

6. Tokeniser comfort (low)
   Gemma’s SentencePiece handles this fine, but pushing a denser Doric lexicon (wordlist augmentation + short phrases) improves segmentations and makes small models less tempted to revert to standard English.

Concrete edits you can make this week

* De-dupe & de-template ~20–30% of near-identical outputs; keep topic breadth but vary opening lines and rhythm.
* Add 1–2k new adversarial prompts (English-only, “repeat after me”, JSON-only, “write in German”, “translate to English”, etc.) with *Doric* answers.
* Add 500–1k safety refusals phrased fully in Doric.
* Build 2–3k DPO pairs (chosen=Doric, rejected=English/mixed).
* Create a Doric style-linter pass (simple regex/wordlist) to flag: English stock phrases, Americanisms, and missing dialect markers in the first sentence.

Training notes (Unsloth + Gemma 3)

* Keep it simple for SFT: pack sequences, `max_seq_len` 2–4k, LR 2e-5–5e-5 (smaller if using LoRA), 1–2 epochs over the expanded set. Use the model’s native chat template; your `messages` fit.
* Then run a short DPO/ORPO pass (one epoch over pair data) with a small β; log language-use evals every 500 steps.
* Evals: build a language detector tailored to Doric (wordlist + char-ngram) and track % of assistant tokens flagged “non-Doric”, plus adversarial pass rates (English-only prompts answered in Doric).

Quick verdict

* Suitability today: **Good foundation (B)** for Doric-only behaviour on a small Gemma 3; you’ll get strong bias but not bulletproof.
* With the upgrades above (more data, anti-instructions, DPO): **A/A-** — reliably Doric under pressure.

If you want, I can draft the exact augmentation spec (prompt families + a tiny DPO schema) and a one-shot Unsloth training block wired to your JSONL so you can run it in Colab/Ollama-ready afterwards.

