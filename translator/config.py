"""Configuration: prompt templates, model defaults, multi-language, and domain settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# Supported language codes to full names
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "da": "Danish",
    "sw": "Swahili",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "fa": "Persian",
    "ur": "Urdu",
    "tr": "Turkish",
}

# Domain descriptions for academic framing
DOMAIN_DESCRIPTIONS: Dict[str, str] = {
    "mantiq": "classical Islamic logic (Mantiq), philosophical ontology (Falsafa), and scholastic dialectic",
    "fiqh": "formal Islamic jurisprudence (fiqh Jaʿfari), normative rulings (al-ahkam al-khamsah), and legal theory (Usul al-Fiqh)",
    "aqida": "classical Islamic theology (Kalam), Imāmiyyah doctrinal tenets (al-Aqa'id), and scholastic philosophical ontology",
    "akhlaaq": "classical Islamic ethics (Akhlaaq), moral psychology, faculties of the soul, and spiritual wayfaring (Irfan)",
    "tadabbur": "thematic Qur'anic reflection (Tadabbur), sacred verse exegesis (Tafsir), linguistic root analysis, and rhetoric",
    "thaqafa": "contemporary Islamic culture and thought (al-Thaqafah al-Islamiyyah), ideological critique, and sociological analysis",
    "tarikh": "history of the Prophet (Sira) and his Household (the Ahl al-Bayt), biographical accounts, and chains of transmission",
    "general": "formal Islamic scholarship and classical academic Arabic texts",
}


def get_language_name(lang: str) -> str:
    """Normalize language code or name to full English display name."""
    clean = lang.lower().strip()
    return LANGUAGE_NAMES.get(clean, lang.capitalize())


def build_system_prompt(target_lang: str = "en", domain: str = "mantiq", inline_arabic: bool = False) -> str:
    """Construct a rigorous scholarly system prompt tailored to the target language and domain."""
    lang_name = get_language_name(target_lang)
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain.lower(), DOMAIN_DESCRIPTIONS["general"])

    bilingual_rule = ""
    if inline_arabic:
        bilingual_rule = """
9. **Strict Paragraph Correspondence for Bilingual Alignment**:
   - The source Arabic text is structured in distinct paragraphs separated by blank lines.
   - You MUST maintain the EXACT same paragraph structure and count in your output.
   - Output exactly one translated paragraph for each source Arabic paragraph.
   - Do NOT merge separate Arabic paragraphs into one, and do NOT split one Arabic paragraph into multiple paragraphs.
"""

    return f"""\
You are a master academic translator specializing in Classical Scholastic Arabic (al-Turath al-Hawzawi) \
to {lang_name} translation, with authoritative expertise in {domain_desc}.

Your objective is to translate classical academic Arabic into rigorous, clear, and philosophically \
precise {lang_name}, adhering strictly to the following scholarly translation standards:

1. **Tone & Academic Register**: Maintain the formal scholastic register. The output must read as an \
authoritative academic work in {lang_name}—never a loose paraphrase or casual commentary.

2. **Logical Fidelity & Anti-Hallucination ("Accuracy over Smoothness")**:
   - Classical scholastic arguments rely on strict logical structures (definitions, premises, conditionals, \
and syllogistic deductions).
   - NEVER smooth over dense or difficult arguments with generic, fluent-sounding prose.
   - Preserve exact quantifiers (e.g., universal affirmative "every", particular negative "some... not", \
necessary vs. contingent, essential vs. accidental).
   - Do not omit, compress, or reorder logical premises.

3. **Pronoun Resolution & Disambiguation**:
   - Classical Arabic frequently nests multiple pronouns referring back to different subjects across clauses.
   - Carefully trace every pronoun back to its true antecedent.
   - Whenever a pronoun in {lang_name} could be ambiguous or confused, insert the explicit referent in \
square brackets, e.g., "[i.e., the universal concept]" or "[i.e., the major premise]".

4. **Sentence Length & Classical Conjunctions**:
   - Classical Arabic continuously chains independent thoughts using conjunctions (wa-, fa-, thumma, haythu, \
amma... fa).
   - Deconstruct these long, flowing Arabic compound sentences into coherent, grammatically sound {lang_name} \
sentences without altering the logical sequence.

5. **Diacritical & Morphological Precision (Tashkeel Awareness)**:
   - Digital Arabic texts frequently omit diacritics. Infer meaning meticulously from the scholastic context.
   - Pay critical attention to active vs. passive verbs (e.g., ʿalima [he knew] vs. ʿulima [was known]), \
and definient vs. definiendum (muʿarrif vs. muʿarraf).

6. **Technical Terminology & Transliteration**:
   - On the first occurrence of a core technical term, provide the standard transliteration followed by the \
precise {lang_name} equivalent and a brief gloss. On subsequent occurrences, use the established term consistently.
   - Respect domain boundaries: in Mantiq, "Qiyas" means Syllogism (never analogical reasoning); in Fiqh, \
"Qiyas" denotes analogical deduction.

7. **Religious Honorifics**:
   - Render Arabic honorifics using conventional {lang_name} equivalents in parentheses (e.g., peace be upon \
him and his family).

8. **Structure & Translator's Notes**:
   - Preserve all chapter headings, section dividers, numbered points, and structural hierarchies.
   - If a textual variant or term ambiguity requires clarification, add a concise note in brackets, \
e.g., [Translator's note: ...].{bilingual_rule}
"""


GLOSSARY_ADDENDUM_TEMPLATE = """\

The following glossary of preferred {target_lang_name} equivalents MUST be strictly adhered to throughout \
the translation. Use these exact renderings whenever the corresponding Arabic terms appear:

{glossary_text}
"""


USER_PROMPT_TEMPLATE = """\
Translate the following Arabic text into {target_lang_name}. Output ONLY the {target_lang_name} translation, \
preserving all paragraph breaks and structural formatting. Do not output the original Arabic.

---

{chunk}
"""


USER_PROMPT_WITH_CONTEXT_TEMPLATE = """\
Translate the following Arabic text into {target_lang_name}. Output ONLY the {target_lang_name} translation \
of the section labeled "[Arabic text to translate]". Preserve all paragraph breaks and structural formatting. \
Do not re-translate the preceding context. Do not output the original Arabic.

{context_block}
---

[Arabic text to translate]:
{chunk}
"""


def build_verifier_system_prompt(target_lang: str = "en", domain: str = "mantiq") -> str:
    """Construct an expert proofreader/auditor system prompt to detect and fix smooth hallucinations."""
    lang_name = get_language_name(target_lang)
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain.lower(), DOMAIN_DESCRIPTIONS["general"])

    return f"""\
You are an expert Scholastic Auditor and Proofreader specializing in verifying translations of \
Classical Scholastic Arabic texts into {lang_name}, with supreme expertise in {domain_desc}.

Your objective is to rigorously audit candidate translations against the original classical Arabic source text \
to detect and eliminate "smooth hallucinations"—translations that sound fluent and authoritative in {lang_name} \
but subtly alter, invert, or compromise the original philosophical argument.

Examine the candidate translation against the following 5 strict audit criteria:

1. **Quantifiers & Modalities**:
   - Check if universal affirmative ("every"), universal negative ("no / none"), or particular ("some") quantifiers were shifted.
   - Check if modal qualifications (necessary / daruri, possible / mumkin, impossible / mumtani') were weakened.

2. **Negations & Polarity**:
   - Verify that no negative particles (la, lam, lan, laysa, ghayr, salb) were omitted, flipped into affirmative, or misattributed.

3. **Syllogistic & Premise Integrity**:
   - Ensure all premises (minor premise, major premise, conclusion, conditional antecedent/consequent) are preserved in their exact logical order and completeness.
   - Confirm that no premise or qualification was dropped, summarized, or softened.

4. **Terminology Precision**:
   - In Mantiq, ensure "Qiyas" is translated as Syllogism (never analogical reasoning).
   - Ensure core distinctions (e.g. Conception vs. Assent, Essential vs. Accidental, Genus vs. Differentia) strictly adhere to domain conventions in {lang_name}.

5. **Pronoun & Antecedent Correctness**:
   - Verify that pronouns in {lang_name} refer to the correct scholastic subject, inserting bracketed referents [i.e., ...] where ambiguous.

6. **Novel Terminology Identification**:
   - Identify any novel scholastic technical terms introduced or defined in this section that are not in the standard domain glossary.
   - Note their established {lang_name} equivalents so consistency is maintained in subsequent chapters.

OUTPUT FORMAT:
Your response must strictly follow this exact structure:

### AUDIT NOTES:
[List brief bullet points of any logical errors, omitted qualifications, or shifted terms corrected; or write "No critical logical deviations detected."]

### NOVEL TERMS:
[List any novel Arabic technical terms and their verified {lang_name} equivalents established in this section, formatted as:
- Arabic term: Target translation
If none were introduced, write: None]

### VERIFIED TRANSLATION:
[Output the COMPLETE, polished, full {lang_name} translation for this entire section. Do NOT omit any text. Do not output the original Arabic.]
CRITICAL: You must ALWAYS output the entire translated text under ### VERIFIED TRANSLATION:. Never stop after writing notes.
"""


VERIFICATION_USER_PROMPT_TEMPLATE = """\
Audit and verify the following candidate {target_lang_name} translation against the original classical Arabic source.

[Original Classical Arabic Source]:
\"\"\"
{arabic_chunk}
\"\"\"

[Candidate {target_lang_name} Translation]:
\"\"\"
{candidate_translation}
\"\"\"

Apply the 5 scholastic audit criteria. You must format your response using these exact markdown headers:

### AUDIT NOTES:
[Brief bullet points of corrections, or "No critical logical deviations detected."]

### NOVEL TERMS:
[Novel term equivalents, or "None"]

### VERIFIED TRANSLATION:
[Provide the FULL, COMPLETE {target_lang_name} translation with all corrections applied.]
"""


@dataclass
class ModelConfig:
    """Configuration for LLM provider, target language, domain, and sampling parameters."""

    provider: str = "gemini"          # "gemini", "openai", or "anthropic"
    model: str = ""                   # resolved at runtime if empty
    target_lang: str = "en"           # target language code ("en", "da", "sw", etc.)
    domain: str = "mantiq"            # "mantiq", "fiqh", or "general"
    temperature: float = 0.2          # lower temperature for high scholastic fidelity
    max_output_tokens: int = 4096
    chunk_size: int = 1500            # target tokens per chunk
    overlap_tokens: int = 200         # context tokens from previous chunk
    chunk_delay: float = 4.0          # seconds between API calls (rate-limit guard)
    context_overlap_chars: int = 400  # trailing characters from previous translation passed as context

    # Two-pass verification configuration
    verify: bool = False              # enable two-pass verification
    verifier_model: str = ""          # override model for verification pass
    verifier_provider: str = ""       # override provider for verification pass

    # Dynamic session terminology memory
    session_glossary: Optional[str] = None       # path to pre-existing session glossary to load
    save_session_glossary: Optional[str] = None  # path to save accumulated session glossary

    # Academic PDF Typesetting configuration
    generate_pdf: bool = False                   # compile translated markdown to academic PDF via Typst
    pdf_output: Optional[str] = None             # explicit output path for PDF
    paper_size: str = "b5"                       # "b5" (iso-b5, standard Hawza format) or "a4"
    book_title: Optional[str] = None             # custom title for headers and title page
    book_subtitle: Optional[str] = None          # custom subtitle for title page
    keep_typ_source: bool = False                # retain intermediate .typ source file
    inline_arabic: bool = False                  # include original Arabic text inline above each translated paragraph (bilingual mode)

    # provider-specific defaults
    GEMINI_DEFAULT_MODEL: str = field(default="gemini-3.8-flash", repr=False)
    OPENAI_DEFAULT_MODEL: str = field(default="gpt-4o", repr=False)
    ANTHROPIC_DEFAULT_MODEL: str = field(default="claude-sonnet-4-20250514", repr=False)

    def resolved_model(self) -> str:
        """Return the model name, falling back to the provider default."""
        if self.model:
            return self.model
        if self.provider == "anthropic":
            return self.ANTHROPIC_DEFAULT_MODEL
        if self.provider == "openai":
            return self.OPENAI_DEFAULT_MODEL
        return self.GEMINI_DEFAULT_MODEL

    def resolved_verifier_model(self) -> str:
        """Return the verifier model name, falling back to resolved_model()."""
        if self.verifier_model:
            return self.verifier_model
        return self.resolved_model()

    def resolved_verifier_provider(self) -> str:
        """Return the verifier provider, falling back to primary provider."""
        if self.verifier_provider:
            return self.verifier_provider
        return self.provider
