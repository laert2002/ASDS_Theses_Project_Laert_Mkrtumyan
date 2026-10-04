"""
Task 3 - Tokenization: English vs Armenian.

What it does
------------
For five English and five Armenian sentences (same meaning, parallel pairs):
  * prints the token IDs,
  * prints each token as the raw BPE piece AND as decoded text,
  * counts words, characters, UTF-8 bytes and tokens,
  * reports tokens-per-word for each language.

Results -> results/tokenization.json ; write-up -> docs/tokenization.md

Run:
    python src/tokenization_demo.py

Notes
-----
* Only the tokenizer is downloaded (a few MB); no model weights and no GPU needed.
* Llama-3.2-1B and Llama-3.1-8B use the SAME tokenizer (128k-vocab byte-level BPE),
  so the numbers apply to both models.
* We use add_special_tokens=False when counting so the BOS token
  <|begin_of_text|> does not inflate the counts.
"""

from utils import load_config, load_tokenizer, save_json

# Parallel sentences: sentence i in English means the same as sentence i in Armenian.
# This makes the comparison fair - same content, different language.
SENTENCES = {
    "english": [
        "Yerevan is the capital of Armenia.",
        "I love reading books.",
        "The weather is very nice today.",
        "We study at the university.",
        "Artificial intelligence is developing rapidly.",
    ],
    "armenian": [
        "Երևանը Հայաստանի մայրաքաղաքն է։",
        "Ես սիրում եմ կարդալ գրքեր։",
        "Այսօր եղանակը շատ լավ է։",
        "Մենք սովորում ենք համալսարանում։",
        "Արհեստական բանականությունը արագ զարգանում է։",
    ],
}


def analyse_sentence(tokenizer, text: str) -> dict:
    ids = tokenizer.encode(text, add_special_tokens=False)

    # convert_ids_to_tokens -> the raw vocabulary entry. Byte-level BPE shows a
    # leading space as "Ġ" and non-ASCII bytes as odd Latin-1 looking symbols.
    pieces = tokenizer.convert_ids_to_tokens(ids)

    # decode([id]) -> the text that single token represents. If a token holds
    # only PART of a multi-byte UTF-8 character you will see "�" here.
    decoded = [tokenizer.decode([i]) for i in ids]

    # Simple whitespace split; punctuation stays attached to the word, which is
    # the usual convention for "tokens per word" ("fertility") measurements.
    n_words = len(text.split())
    return {
        "text": text,
        "token_ids": ids,
        "pieces": pieces,
        "decoded": decoded,
        "n_tokens": len(ids),
        "n_words": n_words,
        "n_chars": len(text),
        "n_bytes": len(text.encode("utf-8")),
        "tokens_per_word": len(ids) / n_words,
    }


def main() -> None:
    cfg = load_config()
    model_id = cfg["models"]["small"]
    tokenizer = load_tokenizer(model_id)
    print(f"Tokenizer: {model_id} | vocab size (incl. special tokens) = {len(tokenizer)}")

    # Show once what the special BOS token looks like.
    example = tokenizer("Hello")["input_ids"]
    print(f"With special tokens, 'Hello' -> {example} -> {tokenizer.convert_ids_to_tokens(example)}\n")

    results, summary = {}, {}
    for lang, sentences in SENTENCES.items():
        print("=" * 70, f"\n{lang.upper()}\n" + "=" * 70)
        rows = [analyse_sentence(tokenizer, s) for s in sentences]
        for r in rows:
            print(f"\n{r['text']}")
            print(f"  ids     : {r['token_ids']}")
            print(f"  pieces  : {r['pieces']}")
            print(f"  decoded : {r['decoded']}")
            print(f"  words={r['n_words']} tokens={r['n_tokens']} "
                  f"chars={r['n_chars']} bytes={r['n_bytes']} tokens/word={r['tokens_per_word']:.2f}")

        total_tokens = sum(r["n_tokens"] for r in rows)
        total_words = sum(r["n_words"] for r in rows)
        total_bytes = sum(r["n_bytes"] for r in rows)
        total_chars = sum(r["n_chars"] for r in rows)
        # Corpus-level ratio (total tokens / total words) is more robust than
        # averaging per-sentence ratios, because short sentences don't dominate.
        summary[lang] = {
            "total_words": total_words,
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "total_bytes": total_bytes,
            "tokens_per_word": total_tokens / total_words,
            "chars_per_token": total_chars / total_tokens,
            "bytes_per_token": total_bytes / total_tokens,
        }
        results[lang] = rows

    print("\n" + "=" * 70 + "\nSUMMARY (paste into docs/tokenization.md)\n")
    print("| Language | Words | Tokens | Tokens/word | Chars/token | Bytes/token |")
    print("|---|---|---|---|---|---|")
    for lang, s in summary.items():
        print(f"| {lang} | {s['total_words']} | {s['total_tokens']} | {s['tokens_per_word']:.2f} "
              f"| {s['chars_per_token']:.2f} | {s['bytes_per_token']:.2f} |")
    ratio = summary["armenian"]["total_tokens"] / summary["english"]["total_tokens"]
    print(f"\nArmenian needs {ratio:.2f}x more tokens than English for the same content.")

    save_json({"model_id": model_id, "summary": summary, "sentences": results,
               "armenian_to_english_token_ratio": ratio}, "tokenization.json")


if __name__ == "__main__":
    main()
