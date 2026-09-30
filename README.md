# 📖 Arabic Book Translator

Translate Classical and Scholastic Arabic books to **English**, **Danish**, **Swahili**, and more — purpose-built for formal Hawza academic texts (Logic / *Mantiq*, Jurisprudence / *Fiqh*, Philosophy / *Falsafa*).

Uses modern LLMs (Gemini, OpenAI, Anthropic) guided by scholastic translation instructions, modular per-language domain glossaries, and sliding antecedent context to preserve logical rigor and eliminate context drift.

---

## ✨ Features

- **Multi-Language Support** — translate directly to English (`en`), Danish (`da`), Swahili (`sw`), and more.
- **Domain-Specific Glossaries** — specialized glossaries for **Logic (*Mantiq*)** and **Jurisprudence (*Fiqh*)** in every supported language. Prevents disastrous category confusions (e.g. translating *Qiyas* as "syllogism" in logic vs. "analogical deduction" in fiqh).
- **Antecedent Context Chaining** — slides context across chunk boundaries so LLMs correctly resolve ambiguous Arabic pronouns and maintain uniform terminology across chapters.
- **Dynamic Session Terminology Memory** — accumulates novel scholastic terms across chunks during a book translation, dynamically injecting them into subsequent chunk prompts to prevent terminology drift on long books.
- **Classical Discourse Chunker** — splits long classical paragraphs on authentic discourse connectors (*Amma ba'd*, *fa-in qila... qulna*, *wa-haythu*, *tanbih*, etc.), not just modern periods.
- **Arabic Text Normalization** — strips tatweel/kashida, normalizes Unicode NFC, and extracts block-sorted text from PDFs to prevent RTL visual scrambling.
- **Scholastic Prompting (Anti-Hallucination)** — primes LLMs for classical scholastic Arabic (*Turathi*), enforcing strict logical fidelity and explicit bracketed referents over superficial English smoothness.
- **Multi-Provider & Resilience** — native support for **Gemini**, **OpenAI**, and **Anthropic** with exponential backoff and rate-limit pacing.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export GOOGLE_API_KEY="your-key-here"

# Translate a logic textbook to English
python -m translator mantiq.pdf --domain mantiq

# Translate to Danish
python -m translator mantiq.pdf --lang da --domain mantiq

# Translate to Swahili
python -m translator mantiq.pdf --lang sw --domain mantiq
```

---

## 📋 Usage

```bash
python -m translator <input> [options]
```

| Option | Short | Description | Default |
|---|---|---|---|
| `input` | | Path to `.txt`, `.pdf`, or `.docx` file | *required* (unless listing glossaries/chapters) |
| `--lang` | `-l` | Target language code (`en`, `da`, `sw`, etc.) | `en` |
| `--domain` | `-d` | Scholarly domain (`mantiq`, `fiqh`, `aqida`, `akhlaaq`, `tadabbur`, `thaqafa`, `tarikh`, `general`) | `mantiq` |
| `--output` | `-o` | Output file path | `<input>_translated[_<lang>].md` |
| `--pages` | | Page range to translate (e.g. `10-25`, `10:`, `:20`, `5`) | all pages |
| `--skip-pages` | | Number of pages to skip from start (e.g. `10`) | `0` |
| `--take-pages` | | Number of pages to translate (e.g. `5`) | all remaining |
| `--chapter` | | Translate single chapter by 1-based index or title query | all |
| `--list-chapters` | | Scan document and list detected chapters & word counts | — |
| `--provider` | | `gemini`, `openai`, or `anthropic` | `gemini` |
| `--model` | | Override provider default model | provider default |
| `--glossary` | | Path to a custom glossary JSON (overrides presets) | built-in by lang/domain |
| `--session-glossary` | | Path to pre-existing session terms JSON to load & enforce | none |
| `--save-session-glossary` | | Save newly discovered session terms to a JSON file | none (or auto `<input>_session_terms_<lang>.json`) |
| `--chunk-size` | | Target tokens per chunk | `1500` |
| `--overlap-tokens` | | Antecedent context tokens from previous chunk | `200` |
| `--verify` | | Enable Two-Pass Scholastic Verification / Self-Critique | `false` |
| `--verifier-model` | | Model override for Pass 2 verification | provider default |
| `--verifier-provider` | | Provider override for Pass 2 (`gemini`, `openai`, `anthropic`) | provider default |
| `--audit-output` | | Path for the verification audit report | `<output>_audit.md` |
| `--pdf` | | Compile translated markdown to publication-ready Hawza PDF via Typst | `false` |
| `--pdf-output` | | Explicit output path for the generated PDF | `<output>.pdf` |
| `--paper-size` | | PDF paper size (`b5` / `iso-b5`, `a4`, `a5`, `letter`) | `b5` |
| `--book-title` | | Custom textbook title for PDF cover page & headers | input filename |
| `--book-subtitle` | | Custom subtitle for PDF cover page | domain preset |
| `--keep-typ` | | Retain intermediate .typ source file alongside PDF | `false` |
| `--keep-headers` | | Keep running headers & page footers in PDFs | `false` |
| `--merge-footnotes` | | Do not isolate bottom footnotes into a separate section | `false` |
| `--temperature`| | Sampling temperature (lower = higher fidelity) | `0.2` |
| `--delay` | | Seconds between API calls (rate-limit guard) | `4.0` |
| `--dry-run` | | Test chunking and pipeline without API calls | — |
| `--list-glossaries` | | List all built-in languages and domain glossaries | — |

---

## 📚 Multi-Language & Domain Glossaries

Inspect all available language and domain combinations anytime:

```bash
python -m translator --list-glossaries
```

Output:
```
📚 Built-in Glossaries by Language & Domain:
  • Danish (`da`): fiqh, mantiq
  • English (`en`): fiqh, mantiq
  • Swahili (`sw`): fiqh, mantiq
```

### Directory Structure

Glossaries are organized cleanly by language code and domain:

```
glossaries/
├── da/
│   ├── fiqh.json
│   └── mantiq.json
├── en/
│   ├── fiqh.json
│   └── mantiq.json
└── sw/
    ├── fiqh.json
    └── mantiq.json
```

### Adding a New Language or Domain

To add support for a new language (e.g. French `fr`, German `de`, or Urdu `ur`), simply create a JSON file at `glossaries/<lang>/<domain>.json`:

```json
{
  "honorifics": {
    "صلى الله عليه وآله وسلم": "(paix et bénédictions sur lui et sa famille)"
  },
  "syllogisms_and_arguments": {
    "القياس": "syllogisme (qiyas)",
    "البرهان": "démonstration (burhan)"
  }
}
```

The translator will automatically detect and load it whenever `--lang fr --domain mantiq` is specified!

---

## 🔄 How It Works

```
┌────────────────────────────────────────────────────────┐
│ 1. Read & Normalize Arabic Text                        │
│    - Strips tatweel (ـ) and cleans Unicode (NFC)        │
│    - Block-sorted PDF extraction (prevents RTL flips)  │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 2. Classical Discourse-Aware Chunker                   │
│    - Breaks on classical connectors (Amma ba'd, etc.)  │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 3. Scholastic Prompt & Domain Glossary Injection       │
│    - Dynamic prompt for target language (EN, DA, SW)   │
│    - Injects domain terms (Mantiq vs. Fiqh)            │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 4. Chunk Translation with Antecedent Context Chaining   │
│    - Passes trailing translation of Chunk N-1 to       │
│      Chunk N for pronoun resolution & term consistency │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 5. Two-Pass Verification Pass (Optional --verify)      │
│    - Audits quantifiers, negations, and premises       │
│    - Eliminates smooth hallucinations & polishes prose │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 6. Output Markdown Writer                              │
│    - Saves to <filename>_translated_<lang>.md          │
└────────────────────────────────────────────────────────┘
```

---

## 🔍 Two-Pass Scholastic Verification (Anti-Hallucination)

Classical logic texts are vulnerable to the *"hallucination of smoothness"*, where an LLM generates English, Danish, or Swahili prose that reads fluently but subtly drops a negative particle, weakens a modal condition (turning *necessary* into *possible*), or flips a quantifier (turning *all* into *some*).

Add `--verify` to run an automated, 5-point audit on every chunk:

```bash
# Translate and audit using Gemini
python -m translator mantiq.pdf --lang da --domain mantiq --verify

# Cross-provider verification: Draft with Gemini, Audit with Claude
python -m translator mantiq.pdf --provider gemini --verify --verifier-provider anthropic --verifier-model claude-sonnet-4-20250514
```

### The 5 Audit Criteria
1. **Quantifiers & Modalities**: Checks whether universal affirmative, universal negative, or modal necessity was altered.
2. **Negations & Polarity**: Verifies that negative particles (`لا`, `لم`, `لن`, `ليس`, `غير`) were not omitted or inverted.
3. **Syllogistic Integrity**: Ensures no premises (minor, major, conclusion) were omitted or condensed.
4. **Terminology Precision**: Confirms scholastic definitions (e.g. *Qiyas* as Syllogism) strictly follow the domain glossary.
5. **Pronoun Correctness**: Verifies bracketed referents `[i.e., ...]` match Arabic antecedents.

### 📋 Post-Run Audit Report & Review (Zero Extra Tokens)
Whenever `--verify` is enabled, the auditor's findings are automatically captured at **zero additional token cost**:
- **Terminal Review Panel**: Displays a concise summary of all scholastic adjustments and novel terms upon completion.
- **Companion Markdown Report (`<output>_audit.md`)**: A detailed audit document containing chunk-by-chunk notes, corrections, and novel terms discovered. Customize the report path with `--audit-output <path>`.

---

## 📑 Slicing & Chapter Selection (`skip.take` & Chapters)

Translating a 400-page book in one go can be slow or costly. You can slice by page ranges or extract specific chapters cleanly without breaking chunking or context chaining:

### 1. Page Range Slicing
Slice by explicit page ranges, open-ended ranges, or LINQ-style skip/take:

```bash
# Translate pages 10 to 25 (1-based, inclusive)
python -m translator mantiq.pdf --pages 10-25

# Skip first 20 pages, then translate 10 pages (skip(20).take(10))
python -m translator mantiq.pdf --skip-pages 20 --take-pages 10

# Translate from page 50 to the end
python -m translator mantiq.pdf --pages 50:

# Translate just the first 15 pages
python -m translator mantiq.pdf --pages :15
```

### 2. Chapter Detection & Translation
Scan the document for classical headings (`الباب الأول`, `الفصل الثاني`, `المقصد الثالث`, `القسم الرابع`, `مبحث ...`, etc.) and translate selected sections:

```bash
# List all detected chapters and their estimated word counts
python -m translator mantiq.pdf --list-chapters

# Translate Chapter 1 only
python -m translator mantiq.pdf --chapter 1 --lang da

# Translate a chapter matching a title substring
python -m translator mantiq.pdf --chapter "مباحث الألفاظ" --lang da
```

Output files automatically incorporate slice tags for easy organization (e.g., `mantiq_p10-25_translated_da.md`, `mantiq_ch1_translated_da.md`).

---

## 🧠 Dynamic Session Terminology Memory (Eliminating Terminology Drift)

On a 200+ page Hawza textbook, an author frequently uses specialized technical terms or novel coinages not present in static domain glossaries (e.g. *al-wujub al-ghayri*, *al-kulli al-mantaqi*, *al-'ilm al-huduri*). 

Without memory, chunk 3 might translate a term as "rational necessity", chunk 25 as "extrinsic requirement", and chunk 80 as "external obligation".

The **Dynamic Session Terminology Memory**:
1. **Automatically Learns**: Accumulates novel terms discovered during Pass 2 self-critique audits or explicit bilingual scholastic parentheticals.
2. **Dynamically Enforces**: Injects newly established terms into the translation prompt of subsequent chunks to guarantee uniform vocabulary across all chapters.
3. **Persists & Reuses**: Allows saving learned terms to JSON and reloading them when translating subsequent chapters or companion volumes.

### Usage Examples

```bash
# Translate a full book with in-memory terminology preservation and auto-save terms
python -m translator mantiq.pdf --lang da --save-session-glossary

# Translate Chapter 1 and export novel terms discovered
python -m translator mantiq.pdf --chapter 1 --lang da --save-session-glossary chapter1_terms.json

# Translate Chapter 2 while reusing Chapter 1's established terminology
python -m translator mantiq.pdf --chapter 2 --lang da --session-glossary chapter1_terms.json --save-session-glossary chapter2_terms.json
```



