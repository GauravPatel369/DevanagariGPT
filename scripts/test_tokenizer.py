"""Test and Analyze SentencePiece Tokenizer for Hindi & Nepali.

Provides qualitative evaluation reports saved into `report/`:
- `report/hindi_tokenizer_test_output.txt`
- `report/nepali_tokenizer_test_output.txt`
- `tokenizer_test_output.txt`

CLI / Interactive selection:
  1: Hindi
  2: Nepali
  3 / both: Both Languages

Usage:
    python scripts/test_tokenizer.py --lang 1
    python scripts/test_tokenizer.py --lang 2 --text "नेपालको संविधानले सबै नागरिकलाई समान अधिकार दिएको छ।"
    python scripts/test_tokenizer.py --interactive
"""

import sys
import os
import argparse
from pathlib import Path
from collections import Counter

os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import sentencepiece as spm
except ImportError:
    print("[ERROR] SentencePiece is not installed. Run: pip install sentencepiece")
    sys.exit(1)

TEST_CASES = {
    "hindi": {
        "good": [
            {
                "title": "01. Compound Economic Term (Named Entity & Postposition Isolation)",
                "text": "भारतीय रिज़र्व बैंक ने डिजिटल मुद्रा प्रणाली की घोषणा की।",
                "expected": "[' भारतीय', ' रिज़र्व', ' बैंक', ' ने', ' डिजिटल', ' मुद्रा', ' प्रणाली', ' की', ' घोषणा', ' की', '।']"
            },
            {
                "title": "02. Complex Derived Polysyllabic Noun (Morphological Stem-Suffix Split)",
                "text": "पर्यावरणविदों ने प्रदूषण नियंत्रण के लिए कड़े कदम उठाने की मांग की।",
                "expected": "[' पर्यावरण', 'विद', 'ों', ' ने', ' प्रदूषण', ' नियंत्रण', ' के', ' लिए', ' कड़े', ' कदम', ' उठाने', ' की', ' मांग', ' की', '।']"
            },
            {
                "title": "03. Literary & Philosophical Compound Nominal",
                "text": "मनुष्य का आत्मबल ही उसकी सबसे बड़ी शक्ति और आत्मनिर्भरता का स्रोत है।",
                "expected": "[' मनुष्य', ' का', ' आत्म', 'बल', ' ही', ' उसकी', ' सबसे', ' बड़ी', ' शक्ति', ' और', ' आत्मनिर्भर', 'ता', ' का', ' स्रोत', ' है', '।']"
            }
        ],
        "bad": [
            {
                "title": "01. Rare Sanskrit Sandhi Over-Segmentation",
                "text": "अत्युत्कृष्ट",
                "expected": "[' अति', ' उत्कृष्ट'] or [' अत्युत्कृष्ट']"
            },
            {
                "title": "02. Foreign English Loanword & Acronym Byte-Fallback",
                "text": "ChatGPT 4.0 AI मॉडल",
                "expected": "[' ChatGPT', ' 4.0', ' AI', ' मॉडल']"
            },
            {
                "title": "03. Nukta / Halant Inconsistent Splitting",
                "text": "ख़ासियत",
                "expected": "[' ख़ासियत']"
            }
        ]
    },
    "nepali": {
        "good": [
            {
                "title": "01. Constitutional & Administrative Statement (Bound Clitic Retention)",
                "text": "नेपालको संविधानले सबै नागरिकलाई समान अधिकार दिएको छ।",
                "expected": "[' नेपाल', 'को', ' संविधान', 'ले', ' सबै', ' नागरिक', 'लाई', ' समान', ' अधिकार', ' दिएको', ' छ', '।']"
            },
            {
                "title": "02. Hyphenated Compound Nominal Preservation",
                "text": "आर्थिक-सामाजिक विकास",
                "expected": "[' आर्थिक', '-', 'सामाजिक', ' विकास']"
            },
            {
                "title": "03. Agglutinative Plural Case Clitic Decomposition",
                "text": "नागरिकहरूलाई मौलिक अधिकारको प्रत्याभूति गरिएको छ।",
                "expected": "[' नागरिक', 'हरूलाई', ' मौलिक', ' अधिकार', 'को', ' प्रत्याभूति', ' गरिएको', ' छ', '।']"
            }
        ],
        "bad": [
            {
                "title": "01. Complex Passive Verbal Inflection Fragmentation",
                "text": "गरिनुपर्छ",
                "expected": "[' गरिनुपर्छ'] or [' गरिनु', 'पर्छ']"
            },
            {
                "title": "02. Polysyllabic Sanskrit Derivative Over-Segmentation",
                "text": "प्रतिपादित",
                "expected": "[' प्रतिपादित']"
            },
            {
                "title": "03. Foreign Technical Loanword Byte-Fallback",
                "text": "OpenAI GPT-4 Turbo र Python प्रविधि",
                "expected": "[' OpenAI', ' GPT-4', ' Turbo', ' र', ' Python', ' प्रविधि']"
            }
        ]
    }
}

def analyze_byte_fallback(tokens):
    ascii_count = 0
    non_ascii_count = 0
    byte_counter = Counter()
    for piece in tokens:
        if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
            val = int(piece[3:-1], 16)
            byte_counter[piece] += 1
            if 0x00 <= val <= 0x7F:
                ascii_count += 1
            else:
                non_ascii_count += 1
    return ascii_count, non_ascii_count, byte_counter

def analyze_case(sp, title: str, text: str, expected: str = None, is_good: bool = True):
    tokens = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)
    words = text.split()
    
    n_chars = len(text)
    n_words = len(words)
    n_toks = len(tokens)
    
    tpw = n_toks / max(1, n_words)
    cpt = n_chars / max(1, n_toks)
    
    unk_id = sp.unk_id()
    unk_count = sum(1 for tid in ids if tid == unk_id)
    unk_rate = 100.0 * unk_count / max(1, n_toks)
    
    ascii_cnt, non_ascii_cnt, byte_ctr = analyze_byte_fallback(tokens)
    tot_fb = ascii_cnt + non_ascii_cnt
    
    lines = []
    lines.append("=" * 92)
    tag = "[GOOD EXAMPLE]" if is_good else "[FAILURE / BAD EXAMPLE]"
    lines.append(f"CASE : {tag} - {title}")
    lines.append("=" * 92)
    lines.append(f"INPUT TEXT               : {text}")
    lines.append(f"ACTUAL TOKENIZED OUTPUT  : {tokens}")
    if expected:
        lines.append(f"EXPECTED SEGMENTATION    : {expected}")
    lines.append(f"TOKEN IDs                : {ids}")
    lines.append(f"METRICS                 : Words={n_words} | Chars={n_chars} | Tokens={n_toks} | Tokens/Word={tpw:.2f} | Chars/Token={cpt:.2f}")
    lines.append(f"UNK STATS               : UNK Count={unk_count} ({unk_rate:.2f}%)")
    lines.append(f"BYTE FALLBACK STATS     : ASCII={ascii_cnt} | Non-ASCII={non_ascii_cnt} | Total={tot_fb}")
    lines.append("=" * 92 + "\n")
    return "\n".join(lines)

def find_best_model(lang: str, model_path: str = None):
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
    tok_dir = ROOT_DIR / lang / "tokenizer"
    models = sorted(tok_dir.glob("*.model"))
    if not models:
        return None
    for pref in ["unigram_10000", "unigram_8000", "bpe_10000", "bpe_8000"]:
        for m in models:
            if pref in m.name:
                return m
    return models[0]

def run_evaluation_for_lang(lang: str, model_path: Path, custom_text: str = None):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    out_lines = []
    header = f"\n{'='*92}\nTOKENIZER EVALUATION & QUALITATIVE ANALYSIS: {lang.upper()}\n   Model File: {model_path}\n   Vocab Size: {sp.get_piece_size():,}\n{'='*92}\n"
    print(header)
    out_lines.append(header)

    if custom_text:
        res = analyze_case(sp, "Custom User Input Statement", custom_text, is_good=True)
        print(res)
        out_lines.append(res)
    else:
        cases = TEST_CASES.get(lang, {})
        print("\n--- [GOOD EXAMPLES] ---\n")
        out_lines.append("\n--- [GOOD EXAMPLES] ---\n")
        for c in cases.get("good", []):
            res = analyze_case(sp, c["title"], c["text"], expected=c.get("expected"), is_good=True)
            print(res)
            out_lines.append(res)

        print("\n--- [FAILURE / BAD EXAMPLES] ---\n")
        out_lines.append("\n--- [FAILURE / BAD EXAMPLES] ---\n")
        for c in cases.get("bad", []):
            res = analyze_case(sp, c["title"], c["text"], expected=c.get("expected"), is_good=False)
            print(res)
            out_lines.append(res)

    lang_report = "\n".join(out_lines)
    
    # Write language-wise report file into report/ directory
    lang_file = REPORT_DIR / f"{lang}_tokenizer_test_output.txt"
    lang_file.write_text(lang_report, encoding="utf-8")
    print(f"[INFO] {lang.capitalize()} Tokenizer Test Report saved to: {lang_file}")

    return lang_report

def parse_lang_arg(arg: str) -> str:
    val = str(arg).strip().lower()
    if val in ["1", "hindi", "hi"]:
        return "hindi"
    elif val in ["2", "nepali", "ne"]:
        return "nepali"
    elif val in ["3", "both", "all"]:
        return "both"
    return "both"

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hindi & Nepali SentencePiece Tokenizer Analysis.")
    parser.add_argument("--lang", default=None, help="Language selection: 1=Hindi, 2=Nepali, 3=Both")
    parser.add_argument("--model-path", type=str, help="Custom path to .model file")
    parser.add_argument("--text", type=str, help="Custom text to tokenize")
    parser.add_argument("--interactive", action="store_true", help="Prompt interactively for input text and language")
    args = parser.parse_args()

    selected_lang = args.lang
    input_text = args.text

    if args.interactive or (args.lang is None and args.text is None and sys.stdin.isatty()):
        print("\n" + "=" * 80)
        print("  INTERACTIVE TOKENIZER EVALUATOR (HINDI / NEPALI)")
        print("=" * 80)
        print("Select Target Language:")
        print("  [1] Hindi (hi)")
        print("  [2] Nepali (ne)")
        print("  [3] Both Languages (default)")
        choice = input("Enter choice [1/2/3]: ").strip()
        selected_lang = parse_lang_arg(choice if choice else "3")

        custom_in = input("\nEnter custom text to tokenize (or press Enter for full test suite): ").strip()
        if custom_in:
            input_text = custom_in
    else:
        selected_lang = parse_lang_arg(args.lang if args.lang else "both")

    full_output = []

    if selected_lang in ["hindi", "both"]:
        m_path = find_best_model("hindi", args.model_path if selected_lang == "hindi" else None)
        if m_path:
            res_hi = run_evaluation_for_lang("hindi", m_path, custom_text=input_text)
            full_output.append(res_hi)

    if selected_lang in ["nepali", "both"]:
        m_path = find_best_model("nepali", args.model_path if selected_lang == "nepali" else None)
        if m_path:
            res_ne = run_evaluation_for_lang("nepali", m_path, custom_text=input_text)
            full_output.append(res_ne)

    
if __name__ == "__main__":
    main()
