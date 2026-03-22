"""Generate golden test cases for the Go SentencePiece port.

Produces:
  _testdata/golden/test_cases.jsonl
  _testdata/golden/model_info.json
"""

import json
import os
import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "spm.model")
GOLDEN_DIR = os.path.join(HERE, "golden")
OUTPUT_JSONL = os.path.join(GOLDEN_DIR, "test_cases.jsonl")
OUTPUT_INFO = os.path.join(GOLDEN_DIR, "model_info.json")


def make_case(sp, text, description):
    """Create a single test case dict."""
    ids = sp.EncodeAsIds(text)
    pieces = sp.EncodeAsPieces(text)
    decoded = sp.DecodeIds(ids)
    return {
        "input": text,
        "pieces": pieces,
        "ids": ids,
        "decoded": decoded,
        "description": description,
    }


def generate_cases(sp):
    """Generate all test cases."""
    cases = []

    def add(text, desc):
        cases.append(make_case(sp, text, desc))

    # ── Basic ────────────────────────────────────────────────
    add("", "empty_string")
    add("a", "single_ascii_char")
    add("Z", "single_ascii_uppercase")
    add("5", "single_digit")
    add("!", "single_punctuation")
    add("я", "single_cyrillic")
    add("中", "single_cjk")
    add("😀", "single_emoji")
    add("hello", "single_word")
    add("Hello", "single_word_capitalized")
    add("HELLO", "single_word_uppercase")
    add("hello world", "two_words")
    add("Hello World", "two_words_capitalized")
    add("the quick brown fox jumps over the lazy dog", "pangram_english")
    add("THE QUICK BROWN FOX", "pangram_upper")

    # Numbers
    add("0", "zero")
    add("42", "integer")
    add("3.14", "float")
    add("-1", "negative_int")
    add("-273.15", "negative_float")
    add("1000000", "large_number")
    add("0.001", "small_float")
    add("1e10", "scientific_notation")
    add("0xFF", "hex_number")
    add("0b1010", "binary_number")

    # Punctuation
    add("...", "ellipsis")
    add("!?!?", "mixed_punctuation")
    add("---", "dashes")
    add("(((", "parens")
    add("[]{}", "brackets")
    add("@#$%^&*", "special_chars")
    add("\"quoted\"", "double_quoted")
    add("'single'", "single_quoted")
    add(",;:.", "delimiters")

    # ── Unicode & Encoding ───────────────────────────────────
    # Latin extended
    add("café", "latin_accent")
    add("naïve", "latin_diaeresis")
    add("résumé", "latin_multiple_accents")
    add("über", "latin_umlaut")
    add("señor", "latin_tilde")

    # Cyrillic
    add("Привіт світ", "ukrainian_hello")
    add("Привет мир", "russian_hello")
    add("Тестування токенізатора", "ukrainian_tokenizer_testing")
    add("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя", "ukrainian_alphabet_lower")
    add("АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ", "ukrainian_alphabet_upper")
    add("Щастя", "ukrainian_word_shcha")

    # CJK
    add("你好世界", "chinese_hello_world")
    add("東京都", "japanese_tokyo")
    add("こんにちは", "japanese_hiragana")
    add("カタカナ", "japanese_katakana")
    add("한국어", "korean")
    add("漢字かなカナmixed", "japanese_mixed_scripts")

    # Arabic
    add("مرحبا بالعالم", "arabic_hello")
    add("العربية", "arabic_word")

    # Thai
    add("สวัสดีครับ", "thai_hello")

    # Devanagari
    add("नमस्ते", "hindi_hello")
    add("हिन्दी", "hindi_word")

    # Mixed scripts
    add("Hello Привіт 你好 🚀", "mixed_scripts")
    add("abc абв 123 ①②③", "mixed_scripts_numbers")
    add("English Українська 日本語 العربية", "four_scripts")

    # Combining characters
    add("é", "precomposed_e_acute")  # U+00E9
    add("é", "decomposed_e_acute")  # e + U+0301 (may look same but different bytes)
    add("ñ", "precomposed_n_tilde")
    add("Ω", "greek_omega")
    add("ℌ", "fraktur_H")  # NFKC normalizes

    # Emoji
    add("🚀", "rocket_emoji")
    add("👍", "thumbs_up")
    add("❤️", "red_heart_with_vs16")
    add("👨‍👩‍👧‍👦", "family_zwj")
    add("👩🏽‍💻", "woman_technologist_medium_skin")
    add("🏳️‍🌈", "rainbow_flag")
    add("🇺🇦", "flag_ukraine")
    add("🇺🇸", "flag_us")
    add("😀😁😂🤣😃😄😅😆", "emoji_sequence")
    add("text 🎉 more text 🎊 end", "emoji_in_text")

    # Unicode special
    add("\u200b", "zero_width_space")
    add("\ufeff", "bom")
    add("\u00ad", "soft_hyphen")
    add("\u200c", "zwnj")
    add("\u200d", "zwj_alone")
    add("\u2028", "line_separator")
    add("\u2029", "paragraph_separator")
    add("\u00a0", "nbsp")
    add("a\u0300", "a_with_combining_grave")
    add("o\u0308", "o_with_combining_diaeresis")

    # ── Whitespace ───────────────────────────────────────────
    add(" ", "single_space")
    add("  ", "double_space")
    add("   ", "triple_space")
    add("hello  world", "double_space_between_words")
    add("hello   world", "triple_space_between_words")
    add(" hello", "leading_space")
    add("hello ", "trailing_space")
    add(" hello ", "leading_and_trailing_space")
    add("  hello  world  ", "multiple_spaces_everywhere")
    add("\t", "tab")
    add("\n", "newline")
    add("\r\n", "crlf")
    add("\r", "carriage_return")
    add("\t\n\r", "mixed_whitespace_chars")
    add("hello\tworld", "tab_between_words")
    add("hello\nworld", "newline_between_words")
    add("line1\nline2\nline3", "multiline")
    add("line1\r\nline2\r\nline3", "multiline_crlf")
    add("\v", "vertical_tab")
    add("\f", "form_feed")

    # Only whitespace
    add("    ", "four_spaces")
    add("\t\t\t", "three_tabs")
    add("\n\n\n", "three_newlines")

    # ── Length ───────────────────────────────────────────────
    add("a" * 100, "repeated_a_100")
    add("a" * 1000, "repeated_a_1000")
    add("hello " * 100, "hello_repeated_100")
    add("word" * 500, "word_repeated_500_no_spaces")
    long_text = "The quick brown fox jumps over the lazy dog. " * 250
    add(long_text, "long_pangram_10k_chars")
    add("x" * 10000, "single_char_10k")
    add("абв" * 1000, "cyrillic_repeated_3k")

    # ── Special patterns ─────────────────────────────────────
    # URLs
    add("https://example.com", "url_simple")
    add("https://example.com/path?q=hello&lang=uk", "url_with_params")
    add("http://www.example.co.uk/page#section", "url_with_fragment")
    add("ftp://files.example.com/doc.pdf", "url_ftp")

    # Email
    add("user@example.com", "email_simple")
    add("first.last+tag@sub.domain.com", "email_complex")

    # Code
    add('func main() { fmt.Println("hello") }', "go_code")
    add("def foo(x): return x * 2", "python_code")
    add("SELECT * FROM users WHERE id = 1;", "sql_code")
    add("console.log('hello');", "js_code")
    add("#include <stdio.h>", "c_include")
    add("if err != nil { return err }", "go_error_handling")
    add("fn main() -> Result<(), Box<dyn Error>>", "rust_code")

    # JSON
    add('{"key": "value", "num": 42}', "json_object")
    add('[1, 2, 3, "four"]', "json_array")
    add('{"nested": {"a": [1, 2]}}', "json_nested")

    # Markdown
    add("# Heading\n\n**bold** and *italic*", "markdown")
    add("- item 1\n- item 2\n- item 3", "markdown_list")
    add("[link](https://example.com)", "markdown_link")
    add("```go\nfmt.Println()\n```", "markdown_code_block")

    # HTML
    add("<p>Hello <b>world</b></p>", "html_simple")
    add('<div class="container">', "html_with_attr")
    add("&amp; &lt; &gt;", "html_entities")
    add("<script>alert('xss')</script>", "html_script")

    # Numbers with formatting
    add("1,234,567.89", "number_formatted")
    add("$1,000.00", "currency")
    add("100%", "percentage")
    add("+1 (555) 123-4567", "phone_number")

    # Dates
    add("2024-01-15", "date_iso")
    add("15/01/2024", "date_slash")
    add("January 15, 2024", "date_long")
    add("15.01.2024", "date_dot")

    # Paths
    add("/usr/local/bin/python3", "unix_path")
    add("C:\\Users\\test\\file.txt", "windows_path")
    add("~/Documents/file.pdf", "home_path")

    # ── Byte fallback ────────────────────────────────────────
    # Rare Unicode that likely triggers byte fallback
    add("\U0001F9FF", "rare_emoji_nazar")
    add("\U00013000", "egyptian_hieroglyph")
    add("\U0001D11E", "musical_symbol_g_clef")
    add("\U0001F600\U0001F601\U0001F602", "emoji_triple")
    add("𐍈", "gothic_letter")  # U+10348
    add("𒀀", "cuneiform_sign")  # U+12000
    add("𝕳𝖊𝖑𝖑𝖔", "math_fraktur_hello")
    add("⿰氵月", "cjk_radical")
    add("\U000E0001", "tag_latin_a")  # language tags

    # Characters that may not be in vocab
    add("℃", "degree_celsius")
    add("™", "trademark")
    add("©", "copyright")
    add("®", "registered")
    add("¶", "pilcrow")
    add("§", "section_sign")
    add("†", "dagger")
    add("‡", "double_dagger")
    add("•", "bullet")
    add("…", "horizontal_ellipsis_char")

    # ── Model-specific ───────────────────────────────────────
    # Repeated characters
    add("aaaaaaa", "repeated_a_7")
    add("абабабаб", "repeated_cyrillic_pattern")
    add("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "repeated_a_30")
    add("zzzzzzzzzz", "repeated_z_10")
    add("!!!!!!", "repeated_exclamation")
    add("......", "repeated_dots")

    # Edge cases for tokenization
    add("I", "single_I")
    add("a", "single_a_lower")
    add(" a", "space_then_a")
    add("  a", "two_spaces_then_a")
    add("a ", "a_then_space")
    add("a  ", "a_then_two_spaces")
    add(" a ", "space_a_space")

    # Sentence boundaries
    add("Hello. World.", "two_sentences")
    add("Hello! World? Yes.", "mixed_sentence_endings")
    add("Mr. Smith went to Washington.", "abbreviation_period")
    add("U.S.A.", "acronym_dots")
    add("etc. and so on", "etc_abbreviation")

    # Case variations
    add("camelCase", "camelCase")
    add("snake_case", "snake_case")
    add("kebab-case", "kebab_case")
    add("PascalCase", "PascalCase")
    add("SCREAMING_SNAKE", "screaming_snake")
    add("mixedCASEtext", "mixed_case")

    # Common subword patterns
    add("unhappiness", "prefix_suffix")
    add("internationalization", "long_word")
    add("antidisestablishmentarianism", "very_long_word")
    add("pneumonoultramicroscopicsilicovolcanoconiosis", "longest_english_word")
    add("tokenization", "word_tokenization")
    add("detokenization", "word_detokenization")
    add("pre-processing", "hyphenated_word")
    add("self-attention", "hyphenated_ml_term")

    # Special tokens context
    add("[CLS]", "cls_token_text")
    add("[SEP]", "sep_token_text")
    add("[MASK]", "mask_token_text")
    add("[UNK]", "unk_token_text")
    add("<s>", "bos_token_text")
    add("</s>", "eos_token_text")
    add("<pad>", "pad_token_text")

    # Math and formulas
    add("x² + y² = z²", "math_squares")
    add("∑(i=1..n) xᵢ", "math_summation")
    add("α β γ δ ε", "greek_letters")
    add("∫₀¹ f(x)dx", "math_integral")
    add("√2 ≈ 1.414", "math_sqrt")
    add("∞", "infinity")
    add("≤ ≥ ≠ ≡", "math_relations")

    # Programming symbols
    add("->", "arrow")
    add("=>", "fat_arrow")
    add(":=", "walrus")
    add("!=", "not_equal")
    add("==", "double_equal")
    add("&&", "logical_and")
    add("||", "logical_or")
    add("<<", "left_shift")
    add(">>", "right_shift")
    add("**", "double_star")
    add("//", "double_slash")

    # Escape sequences as literal text
    add("\\n", "escaped_newline")
    add("\\t", "escaped_tab")
    add("\\\\", "escaped_backslash")
    add('\\"', "escaped_quote")
    add("\\u0041", "escaped_unicode")

    # Real-world text samples
    add("The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017.", "ml_text")
    add("To install, run: pip install sentencepiece>=0.1.99", "install_instruction")
    add("Error: connection refused at 127.0.0.1:8080", "error_message")
    add("2024-01-15T10:30:00Z", "iso_datetime")
    add("Mon Jan 15 10:30:00 UTC 2024", "unix_datetime")
    add("git commit -m 'fix: resolve tokenizer edge case'", "git_command")
    add("SELECT u.name, COUNT(*) as cnt FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name HAVING cnt > 5 ORDER BY cnt DESC;", "complex_sql")
    add("192.168.1.1", "ip_address")
    add("fe80::1%en0", "ipv6_address")
    add("FF:FF:FF:FF:FF:FF", "mac_address")
    add("sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "hash")

    # Ukrainian text samples
    add("Україна — держава у Східній та Центральній Європі.", "ukrainian_sentence")
    add("Київ — столиця і найбільше місто України.", "ukrainian_kyiv")
    add("Слава Україні! Героям слава!", "ukrainian_glory")
    add("Ґанок, ґудзик, ґрунт", "ukrainian_g_letter")
    add("Їжак їсть їжу", "ukrainian_yi")
    add("Це є тестовий текст для перевірки токенізатора.", "ukrainian_test_text")

    # Russian text
    add("Привет, как дела?", "russian_greeting")
    add("Москва — столица Российской Федерации.", "russian_moscow")

    # Mixed language sentences
    add("I love Київ (Kyiv) — it's beautiful!", "mixed_en_uk")
    add("The word 'Привіт' means 'hello' in Ukrainian.", "mixed_en_uk_quotes")
    add("Use `fmt.Println(\"Привіт\")` to print hello.", "code_with_cyrillic")

    # Edge: very short inputs
    add(".", "single_period")
    add(",", "single_comma")
    add(" ", "single_space_2")
    add("\n", "single_newline_2")
    add("ab", "two_chars")
    add("abc", "three_chars")

    # Control characters (within valid text)
    add("hello\x00world", "null_byte")
    add("hello\x01world", "soh_byte")
    add("hello\x7fworld", "del_byte")
    add("test\x1b[31mred\x1b[0m", "ansi_escape")

    # Repetitive patterns
    add("ha" * 50, "haha_100_chars")
    add("abc" * 100, "abc_repeated_300")
    add("12345" * 50, "digits_repeated_250")
    add(",.!? " * 50, "punctuation_repeated")

    # Boundary: empty-ish after normalization
    add("\u200b\u200b\u200b", "multiple_zwsp")
    add("\ufeff\ufeff", "multiple_bom")

    # Tabs and indentation (common in code)
    add("    if True:\n        pass", "python_indented")
    add("\t\tindented", "double_tab_indent")

    # Long words from various languages
    add("Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz", "german_long_word")
    add("Непротивоконституціонерствувати", "ukrainian_long_word")
    add("supercalifragilisticexpialidocious", "english_long_word")

    # Numbers in text
    add("I have 3 cats and 2 dogs.", "numbers_in_sentence")
    add("Chapter 12: The End", "chapter_number")
    add("v2.0.1-beta.3", "version_string")
    add("2^10 = 1024", "power_notation")

    # Quotation styles
    add('"Hello," she said.', "english_quotes")
    add("«Привіт», — сказала вона.", "ukrainian_quotes")
    add("'It\\'s fine,' he replied.", "apostrophe_in_quotes")
    add("\u201eHallo\u201c, sagte er.", "german_quotes")

    # Whitespace-sensitive
    add("a b", "spaced_letters")
    add("a  b  c", "double_spaced_letters")
    add("word1  word2  word3  word4  word5", "double_spaced_words")

    # Currency and symbols
    add("$100 €200 £300 ¥400 ₴500", "currencies")
    add("±0.5", "plus_minus")
    add("°C °F", "degree_symbols")
    add("µm", "micro_prefix")
    add("Ω", "ohm_symbol")

    # Very mixed content
    add("Hello! 你好! مرحبا! Привіт! 🎉", "five_language_greeting")
    add("Price: $19.99 (€18.50, £16.00)", "price_multi_currency")
    add("2024/01/15 10:30 AM — Meeting with Dr. Smith re: project #42", "calendar_entry")
    add("TODO: fix bug #1234 — see https://github.com/org/repo/issues/1234", "todo_with_link")
    add("name=John&age=30&city=Kyiv&lang=uk", "query_string")
    add("Content-Type: application/json; charset=utf-8", "http_header")
    add("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0", "jwt_token")
    add("data:image/png;base64,iVBORw0KGgo=", "data_uri")
    add("user:password@host:5432/database", "connection_string")
    add("cron: 0 */6 * * *", "cron_expression")
    add("regex: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", "regex_pattern")

    # Additional edge cases to reach 500+
    add("" + chr(i) for i in range(32, 127)) if False else None  # skip, we'll do it differently

    # ASCII printable range (batch)
    for i in range(32, 127):
        ch = chr(i)
        add(ch, f"ascii_{i:03d}_{repr(ch)}")

    # Single Unicode chars from various blocks
    for cp, name in [
        (0x00C0, "latin_A_grave"),
        (0x0100, "latin_A_macron"),
        (0x0410, "cyrillic_A"),
        (0x0531, "armenian_Ayb"),
        (0x0621, "arabic_hamza"),
        (0x0E01, "thai_ko_kai"),
        (0x3041, "hiragana_small_a"),
        (0x30A1, "katakana_small_a"),
        (0x4E00, "cjk_unified_one"),
        (0xAC00, "hangul_ga"),
        (0x2200, "for_all"),
        (0x2603, "snowman"),
        (0x2764, "heavy_heart"),
        (0x1F300, "cyclone"),
        (0x1F4A9, "pile_of_poo"),
        (0x1F680, "rocket"),
    ]:
        add(chr(cp), f"char_{name}_U{cp:04X}")

    # Additional cases to reach 500+
    # Multiword sentences in various languages
    add("Доброго ранку! Як справи?", "ukrainian_morning")
    add("Добрий вечір, панове та пані.", "ukrainian_evening")
    add("Сьогодні гарна погода.", "ukrainian_weather")
    add("Програмування — це мистецтво.", "ukrainian_programming")
    add("Тарас Шевченко — великий поет.", "ukrainian_shevchenko")
    add("Bonjour le monde!", "french_hello")
    add("Guten Tag, wie geht es Ihnen?", "german_greeting")
    add("Buenas tardes, señor.", "spanish_greeting")
    add("Buongiorno a tutti!", "italian_greeting")
    add("Olá mundo!", "portuguese_hello")

    # Tricky subword boundaries
    add("unbelievable", "un_believable")
    add("misunderstanding", "mis_understanding")
    add("reintroduction", "re_introduction")
    add("preprocessor", "pre_processor")
    add("postprocessing", "post_processing")
    add("microservices", "micro_services")
    add("multithreading", "multi_threading")
    add("overengineering", "over_engineering")

    # Common abbreviations & acronyms
    add("e.g.", "eg_abbr")
    add("i.e.", "ie_abbr")
    add("Dr. Smith", "dr_abbr")
    add("Mt. Everest", "mt_abbr")
    add("GPU vs CPU", "gpu_vs_cpu")
    add("API", "api_acronym")
    add("HTTP/2", "http2")
    add("OAuth2.0", "oauth2")
    add("JSON-LD", "json_ld")
    add("GraphQL", "graphql")

    # More code patterns
    add("package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"Hello\")\n}", "go_full_program")
    add("from typing import List, Optional, Dict", "python_typing_import")
    add("async function fetchData() { await fetch('/api'); }", "js_async")
    add("@decorator\ndef method(self, arg: str) -> bool:", "python_decorator")
    add("struct Node<T> { value: T, next: Option<Box<Node<T>>> }", "rust_generic_struct")

    # Boundary values
    add(chr(0x7F), "delete_char")
    add(chr(0x80), "first_non_ascii")
    add(chr(0xFF), "latin_small_y_diaeresis")
    add(chr(0x100), "latin_A_macron_char")
    add(chr(0xFFFF), "max_bmp")
    add(chr(0x10000), "first_supplementary")
    add(chr(0x10FFFF), "max_unicode")

    # Strings with mixed newlines and content
    add("first\nsecond\tthird fourth", "mixed_separators_in_text")
    add("one\ttwo\tthree\tfour\tfive", "tab_separated_values")
    add("a,b,c,d,e,f,g,h,i,j", "csv_line")
    add("key1=val1;key2=val2;key3=val3", "semicolon_separated")
    add("path/to/some/deeply/nested/directory/file.txt", "deep_path")

    # More emoji combinations
    add("👋🏻", "waving_hand_light")
    add("👋🏿", "waving_hand_dark")
    add("🧑‍🔬", "scientist")
    add("🏴‍☠️", "pirate_flag")
    add("🫠", "melting_face")
    add("🥹", "holding_back_tears")

    # Sentences with numbers mixed in
    add("I bought 3 apples for $2.50 each.", "shopping_text")
    add("The temperature is -5°C today.", "temperature_text")
    add("Score: 42/100 (42%)", "score_text")
    add("Page 1 of 10", "page_number")
    add("Step 3/7: Configure settings", "step_indicator")

    # Real paragraphs
    add("Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.", "nlp_definition")
    add("The Transformer architecture relies on self-attention mechanisms to process sequential data, eliminating the need for recurrence and convolutions entirely.", "transformer_description")

    # Edge: consecutive special chars
    add("!!!???...", "consecutive_special")
    add("((()))", "nested_parens")
    add("<<<>>>", "angle_brackets")
    add("***___---", "markdown_emphasis_chars")
    add("$$$%%%^^^", "special_triples")

    # More Unicode blocks
    add("ꙮ", "multiocular_o")  # U+A66E
    add("⸘", "gnaborretni")  # U+2E18
    add("⁂", "asterism")  # U+2042
    add("⌘", "command_key")  # U+2318
    add("⌥", "option_key")  # U+2325
    add("⇧", "shift_key")  # U+21E7
    add("⏎", "return_key")  # U+23CE
    add("␣", "open_box")  # U+2423
    add("∅", "empty_set")  # U+2205
    add("∴", "therefore")  # U+2234

    # Ligatures and special forms
    add("ﬁ", "fi_ligature")
    add("ﬂ", "fl_ligature")
    add("Ǆ", "dz_digraph")
    add("ﬀ", "ff_ligature")
    add("æ", "ae_ligature")
    add("œ", "oe_ligature")

    # Additional to reach 500+
    add("supercalifragilistic", "supercali")
    add("Hello, World!", "hello_world_exclaim")
    add("foo bar baz qux", "four_simple_words")
    add("the the the the the", "repeated_the")
    add("     hello     world     ", "heavy_padding")
    add("test123test456test", "alphanumeric_mix")
    add("CamelCaseIdentifierName", "long_camel_case")
    add("UPPER lower MiXeD", "case_variations")
    add("a1b2c3d4e5f6g7h8i9j0", "interleaved_alphanum")
    add("This is a sentence with exactly ten words in it.", "ten_word_sentence")
    add("one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten", "ten_lines")
    add("a" * 5000, "repeated_a_5000")
    add("🔥" * 100, "fire_emoji_100")
    add("Hello" * 200, "hello_repeated_200_no_space")
    add(" ".join(["word"] * 100), "hundred_words")

    return cases


def generate_model_info(sp):
    """Extract model metadata."""
    # Get sample vocab entries
    sample_entries = []
    for i in range(min(20, sp.GetPieceSize())):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)
        # Determine type
        if sp.IsUnknown(i):
            ptype = "UNKNOWN"
        elif sp.IsControl(i):
            ptype = "CONTROL"
        elif sp.IsByte(i):
            ptype = "BYTE"
        elif sp.IsUnused(i):
            ptype = "UNUSED"
        else:
            ptype = "NORMAL"
        sample_entries.append({"piece": piece, "score": score, "type": ptype, "id": i})

    # Also get some byte tokens
    byte_samples = []
    for i in range(sp.GetPieceSize()):
        if sp.IsByte(i):
            byte_samples.append({
                "piece": sp.IdToPiece(i),
                "score": sp.GetScore(i),
                "type": "BYTE",
                "id": i,
            })
            if len(byte_samples) >= 5:
                break

    # Some normal tokens
    normal_samples = []
    for i in range(sp.GetPieceSize()):
        if not sp.IsUnknown(i) and not sp.IsControl(i) and not sp.IsByte(i) and not sp.IsUnused(i):
            normal_samples.append({
                "piece": sp.IdToPiece(i),
                "score": sp.GetScore(i),
                "type": "NORMAL",
                "id": i,
            })
            if len(normal_samples) >= 10:
                break

    info = {
        "vocab_size": sp.GetPieceSize(),
        "model_type": "unigram",
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "unk_id": sp.unk_id(),
        "pad_id": sp.pad_id(),
        "byte_fallback": any(sp.IsByte(i) for i in range(sp.GetPieceSize())),
        "first_20_tokens": sample_entries,
        "sample_byte_tokens": byte_samples,
        "sample_normal_tokens": normal_samples,
    }
    return info


def validate_golden(sp, cases):
    """Self-check: verify every case against sentencepiece."""
    errors = 0
    for i, tc in enumerate(cases):
        ids = sp.EncodeAsIds(tc["input"])
        pieces = sp.EncodeAsPieces(tc["input"])
        decoded = sp.DecodeIds(ids)
        if ids != tc["ids"]:
            print(f"MISMATCH ids at {i} ({tc['description']}): {ids} != {tc['ids']}")
            errors += 1
        if pieces != tc["pieces"]:
            print(f"MISMATCH pieces at {i} ({tc['description']})")
            errors += 1
        if decoded != tc["decoded"]:
            print(f"MISMATCH decoded at {i} ({tc['description']}): {repr(decoded)} != {repr(tc['decoded'])}")
            errors += 1
    return errors


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL_PATH)
    print(f"Loaded model: vocab_size={sp.GetPieceSize()}")

    # Generate cases
    cases = generate_cases(sp)
    print(f"Generated {len(cases)} test cases")

    # Validate
    print("Validating...")
    errors = validate_golden(sp, cases)
    if errors > 0:
        print(f"FATAL: {errors} validation errors!")
        raise SystemExit(1)
    print("All cases validated ✓")

    # Write JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for tc in cases:
            f.write(json.dumps(tc, ensure_ascii=False) + "\n")
    print(f"Written {len(cases)} cases to {OUTPUT_JSONL}")

    # Write model info
    info = generate_model_info(sp)
    with open(OUTPUT_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Written model info to {OUTPUT_INFO}")


if __name__ == "__main__":
    main()
