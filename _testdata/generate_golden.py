"""Generate golden test cases for the Go SentencePiece port.

Produces:
  _testdata/golden/test_cases.jsonl
  _testdata/golden/model_info.json

Target: 5000+ test cases with comprehensive coverage.
"""

import json
import os
import random
import string
import unicodedata
import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "spm.model")
GOLDEN_DIR = os.path.join(HERE, "golden")
OUTPUT_JSONL = os.path.join(GOLDEN_DIR, "test_cases.jsonl")
OUTPUT_INFO = os.path.join(GOLDEN_DIR, "model_info.json")

# Fixed seed for reproducibility
random.seed(42)


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
    seen_inputs = set()

    def add(text, desc):
        if text in seen_inputs:
            return
        seen_inputs.add(text)
        cases.append(make_case(sp, text, desc))

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: ORIGINAL HAND-CRAFTED CASES (~500)
    # ═══════════════════════════════════════════════════════════

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

    # Arabic / Thai / Devanagari
    add("مرحبا بالعالم", "arabic_hello")
    add("العربية", "arabic_word")
    add("สวัสดีครับ", "thai_hello")
    add("नमस्ते", "hindi_hello")
    add("हिन्दी", "hindi_word")

    # Mixed scripts
    add("Hello Привіт 你好 🚀", "mixed_scripts")
    add("abc абв 123 ①②③", "mixed_scripts_numbers")
    add("English Українська 日本語 العربية", "four_scripts")

    # Combining characters
    add("é", "precomposed_e_acute")
    add("e\u0301", "decomposed_e_acute")
    add("ñ", "precomposed_n_tilde")
    add("Ω", "greek_omega")
    add("ℌ", "fraktur_H")

    # Emoji
    add("🚀", "rocket_emoji")
    add("👍", "thumbs_up")
    add("❤️", "red_heart_with_vs16")
    add("👨\u200d👩\u200d👧\u200d👦", "family_zwj")
    add("👩🏽\u200d💻", "woman_technologist_medium_skin")
    add("🏳️\u200d🌈", "rainbow_flag")
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
    add("https://example.com", "url_simple")
    add("https://example.com/path?q=hello&lang=uk", "url_with_params")
    add("http://www.example.co.uk/page#section", "url_with_fragment")
    add("ftp://files.example.com/doc.pdf", "url_ftp")
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

    # JSON / Markdown / HTML
    add('{"key": "value", "num": 42}', "json_object")
    add('[1, 2, 3, "four"]', "json_array")
    add('{"nested": {"a": [1, 2]}}', "json_nested")
    add("# Heading\n\n**bold** and *italic*", "markdown")
    add("- item 1\n- item 2\n- item 3", "markdown_list")
    add("[link](https://example.com)", "markdown_link")
    add("```go\nfmt.Println()\n```", "markdown_code_block")
    add("<p>Hello <b>world</b></p>", "html_simple")
    add('<div class="container">', "html_with_attr")
    add("&amp; &lt; &gt;", "html_entities")
    add("<script>alert('xss')</script>", "html_script")

    # Numbers with formatting
    add("1,234,567.89", "number_formatted")
    add("$1,000.00", "currency")
    add("100%", "percentage")
    add("+1 (555) 123-4567", "phone_number")
    add("2024-01-15", "date_iso")
    add("15/01/2024", "date_slash")
    add("January 15, 2024", "date_long")
    add("15.01.2024", "date_dot")

    # Paths
    add("/usr/local/bin/python3", "unix_path")
    add("C:\\Users\\test\\file.txt", "windows_path")
    add("~/Documents/file.pdf", "home_path")

    # ── Byte fallback ────────────────────────────────────────
    add("\U0001F9FF", "rare_emoji_nazar")
    add("\U00013000", "egyptian_hieroglyph")
    add("\U0001D11E", "musical_symbol_g_clef")
    add("\U0001F600\U0001F601\U0001F602", "emoji_triple")
    add("𐍈", "gothic_letter")
    add("𒀀", "cuneiform_sign")
    add("𝕳𝖊𝖑𝖑𝖔", "math_fraktur_hello")
    add("⿰氵月", "cjk_radical")
    add("\U000E0001", "tag_latin_a")

    # Symbols
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
    add("aaaaaaa", "repeated_a_7")
    add("абабабаб", "repeated_cyrillic_pattern")
    add("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "repeated_a_30")
    add("zzzzzzzzzz", "repeated_z_10")
    add("!!!!!!", "repeated_exclamation")
    add("......", "repeated_dots")

    add("I", "single_I")
    add(" a", "space_then_a")
    add("  a", "two_spaces_then_a")
    add("a ", "a_then_space")
    add("a  ", "a_then_two_spaces")
    add(" a ", "space_a_space")

    add("Hello. World.", "two_sentences")
    add("Hello! World? Yes.", "mixed_sentence_endings")
    add("Mr. Smith went to Washington.", "abbreviation_period")
    add("U.S.A.", "acronym_dots")
    add("etc. and so on", "etc_abbreviation")

    add("camelCase", "camelCase")
    add("snake_case", "snake_case")
    add("kebab-case", "kebab_case")
    add("PascalCase", "PascalCase")
    add("SCREAMING_SNAKE", "screaming_snake")
    add("mixedCASEtext", "mixed_case")

    add("unhappiness", "prefix_suffix")
    add("internationalization", "long_word")
    add("antidisestablishmentarianism", "very_long_word")
    add("pneumonoultramicroscopicsilicovolcanoconiosis", "longest_english_word")
    add("tokenization", "word_tokenization")
    add("detokenization", "word_detokenization")
    add("pre-processing", "hyphenated_word")
    add("self-attention", "hyphenated_ml_term")

    add("[CLS]", "cls_token_text")
    add("[SEP]", "sep_token_text")
    add("[MASK]", "mask_token_text")
    add("[UNK]", "unk_token_text")
    add("<s>", "bos_token_text")
    add("</s>", "eos_token_text")
    add("<pad>", "pad_token_text")

    add("x² + y² = z²", "math_squares")
    add("∑(i=1..n) xᵢ", "math_summation")
    add("α β γ δ ε", "greek_letters")
    add("∫₀¹ f(x)dx", "math_integral")
    add("√2 ≈ 1.414", "math_sqrt")
    add("∞", "infinity")
    add("≤ ≥ ≠ ≡", "math_relations")

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

    add("\\n", "escaped_newline")
    add("\\t", "escaped_tab")
    add("\\\\", "escaped_backslash")
    add('\\"', "escaped_quote")
    add("\\u0041", "escaped_unicode")

    # Real-world text
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

    # Ukrainian text
    add("Україна — держава у Східній та Центральній Європі.", "ukrainian_sentence")
    add("Київ — столиця і найбільше місто України.", "ukrainian_kyiv")
    add("Слава Україні! Героям слава!", "ukrainian_glory")
    add("Ґанок, ґудзик, ґрунт", "ukrainian_g_letter")
    add("Їжак їсть їжу", "ukrainian_yi")
    add("Це є тестовий текст для перевірки токенізатора.", "ukrainian_test_text")
    add("Привет, как дела?", "russian_greeting")
    add("Москва — столица Российской Федерации.", "russian_moscow")

    # Mixed language
    add("I love Київ (Kyiv) — it's beautiful!", "mixed_en_uk")
    add("The word 'Привіт' means 'hello' in Ukrainian.", "mixed_en_uk_quotes")
    add("Use `fmt.Println(\"Привіт\")` to print hello.", "code_with_cyrillic")

    # Edge: very short
    add(".", "single_period")
    add(",", "single_comma")
    add("ab", "two_chars")
    add("abc", "three_chars")

    # Control characters
    add("hello\x00world", "null_byte")
    add("hello\x01world", "soh_byte")
    add("hello\x7fworld", "del_byte")
    add("test\x1b[31mred\x1b[0m", "ansi_escape")

    # Repetitive patterns
    add("ha" * 50, "haha_100_chars")
    add("abc" * 100, "abc_repeated_300")
    add("12345" * 50, "digits_repeated_250")
    add(",.!? " * 50, "punctuation_repeated")

    add("\u200b\u200b\u200b", "multiple_zwsp")
    add("\ufeff\ufeff", "multiple_bom")

    add("    if True:\n        pass", "python_indented")
    add("\t\tindented", "double_tab_indent")

    add("Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz", "german_long_word")
    add("Непротивоконституціонерствувати", "ukrainian_long_word")
    add("supercalifragilisticexpialidocious", "english_long_word")

    add("I have 3 cats and 2 dogs.", "numbers_in_sentence")
    add("Chapter 12: The End", "chapter_number")
    add("v2.0.1-beta.3", "version_string")
    add("2^10 = 1024", "power_notation")

    add('"Hello," she said.', "english_quotes")
    add("«Привіт», — сказала вона.", "ukrainian_quotes")
    add("'It\\'s fine,' he replied.", "apostrophe_in_quotes")
    add("\u201eHallo\u201c, sagte er.", "german_quotes")

    add("a b", "spaced_letters")
    add("a  b  c", "double_spaced_letters")
    add("word1  word2  word3  word4  word5", "double_spaced_words")

    add("$100 €200 £300 ¥400 ₴500", "currencies")
    add("±0.5", "plus_minus")
    add("°C °F", "degree_symbols")
    add("µm", "micro_prefix")
    add("Ω", "ohm_symbol")

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

    # ASCII printable range
    for i in range(32, 127):
        add(chr(i), f"ascii_{i:03d}")

    # Single Unicode chars from various blocks
    for cp, name in [
        (0x00C0, "latin_A_grave"), (0x0100, "latin_A_macron"),
        (0x0410, "cyrillic_A"), (0x0531, "armenian_Ayb"),
        (0x0621, "arabic_hamza"), (0x0E01, "thai_ko_kai"),
        (0x3041, "hiragana_small_a"), (0x30A1, "katakana_small_a"),
        (0x4E00, "cjk_unified_one"), (0xAC00, "hangul_ga"),
        (0x2200, "for_all"), (0x2603, "snowman"),
        (0x2764, "heavy_heart"), (0x1F300, "cyclone"),
        (0x1F4A9, "pile_of_poo"), (0x1F680, "rocket"),
    ]:
        add(chr(cp), f"char_{name}_U{cp:04X}")

    # More languages
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

    # Subword boundaries
    add("unbelievable", "un_believable")
    add("misunderstanding", "mis_understanding")
    add("reintroduction", "re_introduction")
    add("preprocessor", "pre_processor")
    add("postprocessing", "post_processing")
    add("microservices", "micro_services")
    add("multithreading", "multi_threading")
    add("overengineering", "over_engineering")

    # Abbreviations
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

    # Code patterns
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

    add("first\nsecond\tthird fourth", "mixed_separators_in_text")
    add("one\ttwo\tthree\tfour\tfive", "tab_separated_values")
    add("a,b,c,d,e,f,g,h,i,j", "csv_line")
    add("key1=val1;key2=val2;key3=val3", "semicolon_separated")
    add("path/to/some/deeply/nested/directory/file.txt", "deep_path")

    # More emoji
    add("👋🏻", "waving_hand_light")
    add("👋🏿", "waving_hand_dark")
    add("🧑\u200d🔬", "scientist")
    add("🏴\u200d☠️", "pirate_flag")
    add("🫠", "melting_face")
    add("🥹", "holding_back_tears")

    add("I bought 3 apples for $2.50 each.", "shopping_text")
    add("The temperature is -5°C today.", "temperature_text")
    add("Score: 42/100 (42%)", "score_text")
    add("Page 1 of 10", "page_number")
    add("Step 3/7: Configure settings", "step_indicator")

    add("Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.", "nlp_definition")
    add("The Transformer architecture relies on self-attention mechanisms to process sequential data, eliminating the need for recurrence and convolutions entirely.", "transformer_description")

    add("!!!???...", "consecutive_special")
    add("((()))", "nested_parens")
    add("<<<>>>", "angle_brackets")
    add("***___---", "markdown_emphasis_chars")
    add("$$$%%%^^^", "special_triples")

    add("ꙮ", "multiocular_o")
    add("⸘", "gnaborretni")
    add("⁂", "asterism")
    add("⌘", "command_key")
    add("⌥", "option_key")
    add("⇧", "shift_key")
    add("⏎", "return_key")
    add("␣", "open_box")
    add("∅", "empty_set")
    add("∴", "therefore")

    add("ﬁ", "fi_ligature")
    add("ﬂ", "fl_ligature")
    add("Ǆ", "dz_digraph")
    add("ﬀ", "ff_ligature")
    add("æ", "ae_ligature")
    add("œ", "oe_ligature")

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

    print(f"  After section 1 (hand-crafted): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: SYSTEMATIC UNICODE COVERAGE (~1500)
    # ═══════════════════════════════════════════════════════════

    # Every Unicode block - single representative char
    unicode_blocks = [
        # Basic Multilingual Plane
        (0x0041, 0x007A, "basic_latin"),
        (0x00C0, 0x00FF, "latin_supplement"),
        (0x0100, 0x017F, "latin_extended_a"),
        (0x0180, 0x024F, "latin_extended_b"),
        (0x0250, 0x02AF, "ipa_extensions"),
        (0x0300, 0x036F, "combining_diacritics"),
        (0x0370, 0x03FF, "greek_coptic"),
        (0x0400, 0x04FF, "cyrillic"),
        (0x0500, 0x052F, "cyrillic_supplement"),
        (0x0530, 0x058F, "armenian"),
        (0x0590, 0x05FF, "hebrew"),
        (0x0600, 0x06FF, "arabic"),
        (0x0900, 0x097F, "devanagari"),
        (0x0980, 0x09FF, "bengali"),
        (0x0A00, 0x0A7F, "gurmukhi"),
        (0x0A80, 0x0AFF, "gujarati"),
        (0x0B00, 0x0B7F, "oriya"),
        (0x0B80, 0x0BFF, "tamil"),
        (0x0C00, 0x0C7F, "telugu"),
        (0x0C80, 0x0CFF, "kannada"),
        (0x0D00, 0x0D7F, "malayalam"),
        (0x0E00, 0x0E7F, "thai"),
        (0x0E80, 0x0EFF, "lao"),
        (0x0F00, 0x0FFF, "tibetan"),
        (0x1000, 0x109F, "myanmar"),
        (0x10A0, 0x10FF, "georgian"),
        (0x1100, 0x11FF, "hangul_jamo"),
        (0x1200, 0x137F, "ethiopic"),
        (0x13A0, 0x13FF, "cherokee"),
        (0x1400, 0x167F, "unified_canadian_aboriginal"),
        (0x1680, 0x169F, "ogham"),
        (0x16A0, 0x16FF, "runic"),
        (0x1700, 0x171F, "tagalog"),
        (0x1780, 0x17FF, "khmer"),
        (0x1800, 0x18AF, "mongolian"),
        (0x1E00, 0x1EFF, "latin_extended_additional"),
        (0x1F00, 0x1FFF, "greek_extended"),
        (0x2000, 0x206F, "general_punctuation"),
        (0x2070, 0x209F, "superscripts_subscripts"),
        (0x20A0, 0x20CF, "currency_symbols"),
        (0x2100, 0x214F, "letterlike_symbols"),
        (0x2150, 0x218F, "number_forms"),
        (0x2190, 0x21FF, "arrows"),
        (0x2200, 0x22FF, "math_operators"),
        (0x2300, 0x23FF, "misc_technical"),
        (0x2460, 0x24FF, "enclosed_alphanumerics"),
        (0x2500, 0x257F, "box_drawing"),
        (0x2580, 0x259F, "block_elements"),
        (0x25A0, 0x25FF, "geometric_shapes"),
        (0x2600, 0x26FF, "misc_symbols"),
        (0x2700, 0x27BF, "dingbats"),
        (0x2800, 0x28FF, "braille"),
        (0x2E80, 0x2EFF, "cjk_radicals_supplement"),
        (0x3000, 0x303F, "cjk_symbols"),
        (0x3040, 0x309F, "hiragana"),
        (0x30A0, 0x30FF, "katakana"),
        (0x3100, 0x312F, "bopomofo"),
        (0x3130, 0x318F, "hangul_compat_jamo"),
        (0x31F0, 0x31FF, "katakana_phonetic"),
        (0x3200, 0x32FF, "enclosed_cjk"),
        (0x3300, 0x33FF, "cjk_compatibility"),
        (0x4E00, 0x9FFF, "cjk_unified"),
        (0xA000, 0xA48F, "yi_syllables"),
        (0xAC00, 0xD7AF, "hangul_syllables"),
        (0xF900, 0xFAFF, "cjk_compat_ideographs"),
        (0xFB00, 0xFB06, "alpha_presentation_a"),
        (0xFE30, 0xFE4F, "cjk_compat_forms"),
        (0xFF00, 0xFFEF, "halfwidth_fullwidth"),
        # SMP (supplementary)
        (0x10000, 0x1007F, "linear_b_syllabary"),
        (0x10080, 0x100FF, "linear_b_ideograms"),
        (0x10300, 0x1032F, "old_italic"),
        (0x10330, 0x1034F, "gothic"),
        (0x10400, 0x1044F, "deseret"),
        (0x1D100, 0x1D1FF, "musical_symbols"),
        (0x1D400, 0x1D7FF, "math_alphanumeric"),
        (0x1F300, 0x1F5FF, "misc_symbols_pictographs"),
        (0x1F600, 0x1F64F, "emoticons"),
        (0x1F680, 0x1F6FF, "transport_map_symbols"),
        (0x1F900, 0x1F9FF, "supplemental_symbols"),
    ]
    for start, end, block_name in unicode_blocks:
        # Sample a few chars from each block
        step = max(1, (end - start) // 5)
        for idx, cp in enumerate(range(start, end + 1, step)):
            try:
                ch = chr(cp)
                # Skip surrogates
                if 0xD800 <= cp <= 0xDFFF:
                    continue
                add(ch, f"ublock_{block_name}_{idx}_U{cp:04X}")
            except (ValueError, OverflowError):
                pass

    print(f"  After section 2 (unicode blocks): {len(cases)} cases")

    # ── Combining character sequences ───────────────────────
    bases = "aeiouAEIOUбвгдкнп"
    combiners = [
        "\u0300", "\u0301", "\u0302", "\u0303", "\u0304", "\u0305",
        "\u0306", "\u0307", "\u0308", "\u0309", "\u030A", "\u030B",
        "\u030C", "\u030D", "\u030F", "\u0310", "\u0311", "\u0312",
        "\u0327", "\u0328", "\u0330", "\u0331",
    ]
    for i, base in enumerate(bases):
        for j, comb in enumerate(combiners):
            add(base + comb, f"combining_{i}_{j}_U{ord(comb):04X}")
        # Double combining
        if i < 5:
            add(base + combiners[0] + combiners[1], f"double_combining_{i}")
            add(base + combiners[2] + combiners[7], f"double_combining_{i}b")

    print(f"  After combining chars: {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: SYSTEMATIC WHITESPACE & BOUNDARY (~200)
    # ═══════════════════════════════════════════════════════════

    # All Unicode whitespace characters
    whitespace_chars = [
        "\u0009", "\u000A", "\u000B", "\u000C", "\u000D",  # ASCII
        "\u0020",  # space
        "\u0085",  # NEL
        "\u00A0",  # NBSP
        "\u1680",  # ogham space
        "\u2000", "\u2001", "\u2002", "\u2003", "\u2004",  # en/em quads
        "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200A",
        "\u2028", "\u2029",  # line/paragraph separators
        "\u202F",  # narrow NBSP
        "\u205F",  # medium math space
        "\u3000",  # ideographic space
    ]
    for i, ws in enumerate(whitespace_chars):
        add(ws, f"ws_char_{i}_U{ord(ws):04X}")
        add(f"hello{ws}world", f"ws_between_{i}_U{ord(ws):04X}")
        add(f"{ws}hello{ws}", f"ws_around_{i}_U{ord(ws):04X}")

    # Mixed whitespace
    for i in range(20):
        chars = random.choices(whitespace_chars, k=random.randint(2, 6))
        add("hello" + "".join(chars) + "world", f"ws_mixed_{i}")

    # Whitespace-only strings of various lengths
    for n in [1, 2, 3, 5, 10, 20, 50]:
        add(" " * n, f"spaces_only_{n}")
        add("\t" * n, f"tabs_only_{n}")

    print(f"  After section 3 (whitespace): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: SYSTEMATIC LENGTH VARIATION (~300)
    # ═══════════════════════════════════════════════════════════

    # Single char repeated at various lengths
    for ch, name in [("a", "a"), ("x", "x"), ("я", "ya"), ("中", "zhong"), ("🔥", "fire")]:
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 200, 500]:
            add(ch * n, f"repeat_{name}_{n}")

    # Words repeated
    for word, name in [("hello ", "hello_sp"), ("test", "test"), ("abc ", "abc_sp")]:
        for n in [1, 5, 10, 25, 50, 100, 200]:
            add(word * n, f"repeat_word_{name}_{n}")

    # Specific lengths around common boundaries
    base_text = "The quick brown fox jumps over the lazy dog"
    for length in [1, 2, 3, 4, 5, 10, 16, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 2000, 4096]:
        text = (base_text + " ") * (length // len(base_text) + 1)
        add(text[:length], f"len_exact_{length}")

    print(f"  After section 4 (length variation): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: REAL-WORLD TEXT PATTERNS (~500)
    # ═══════════════════════════════════════════════════════════

    # English sentences
    english_sentences = [
        "The cat sat on the mat.",
        "It was the best of times, it was the worst of times.",
        "To be, or not to be, that is the question.",
        "All that glitters is not gold.",
        "A journey of a thousand miles begins with a single step.",
        "In the beginning was the Word.",
        "Elementary, my dear Watson.",
        "I think, therefore I am.",
        "That's one small step for man, one giant leap for mankind.",
        "Houston, we have a problem.",
        "May the Force be with you.",
        "You can't handle the truth!",
        "Life is like a box of chocolates.",
        "I'll be back.",
        "Here's looking at you, kid.",
        "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you.",
        "We hold these truths to be self-evident.",
        "It is a truth universally acknowledged.",
        "Call me Ishmael.",
    ]
    for i, sent in enumerate(english_sentences):
        add(sent, f"en_sentence_{i}")
        add(sent.upper(), f"en_sentence_{i}_upper")
        add(sent.lower(), f"en_sentence_{i}_lower")

    # Ukrainian sentences
    uk_sentences = [
        "Шевченко є одним із найвидатніших українських поетів.",
        "Дніпро — одна з найбільших річок Європи.",
        "Карпати — гірська система в Центральній Європі.",
        "Борщ — традиційна українська страва.",
        "Львів — культурна столиця України.",
        "Одеса — місто біля Чорного моря.",
        "Запоріжжя відоме своєю козацькою історією.",
        "Харків — друге за розміром місто України.",
        "Чорнобиль — місце найбільшої ядерної аварії.",
        "Тризуб — герб України.",
        "Вишиванка — традиційна українська сорочка.",
        "Калина — символ України.",
        "Бандура — український народний інструмент.",
        "Козаки боролися за свободу України.",
        "Конституція України була прийнята 28 червня 1996 року.",
        "Гривня — грошова одиниця України.",
        "Києво-Печерська лавра — відома пам'ятка.",
        "Софія Київська побудована в XI столітті.",
        "Українська мова належить до слов'янської групи.",
        "Степан Бандера — суперечлива історична постать.",
    ]
    for i, sent in enumerate(uk_sentences):
        add(sent, f"uk_sentence_{i}")

    # German
    de_sentences = [
        "Ich bin ein Berliner.",
        "Die Würde des Menschen ist unantastbar.",
        "Alle Menschen werden Brüder.",
        "Wer reitet so spät durch Nacht und Wind?",
        "Ö, ü, ä sind deutsche Umlaute.",
        "Der Straßenbahnführer fährt die Straßenbahn.",
    ]
    for i, sent in enumerate(de_sentences):
        add(sent, f"de_sentence_{i}")

    # French
    fr_sentences = [
        "Je pense, donc je suis.",
        "L'État, c'est moi.",
        "Liberté, égalité, fraternité.",
        "À la recherche du temps perdu.",
        "Les Misérables de Victor Hugo.",
        "C'est la vie!",
    ]
    for i, sent in enumerate(fr_sentences):
        add(sent, f"fr_sentence_{i}")

    # Japanese
    jp_sentences = [
        "吾輩は猫である。",
        "東京は日本の首都です。",
        "桜の花が咲いています。",
        "日本語と英語の混合テスト: Hello World",
        "カタカナとひらがなの混合テスト",
        "漢字、カタカナ、ひらがな、ローマ字、数字123",
    ]
    for i, sent in enumerate(jp_sentences):
        add(sent, f"jp_sentence_{i}")

    # Chinese
    zh_sentences = [
        "中华人民共和国万岁。",
        "天行健，君子以自强不息。",
        "学而时习之，不亦说乎。",
        "北京是中国的首都。",
        "上海是中国最大的城市。",
        "长城是世界文化遗产。",
    ]
    for i, sent in enumerate(zh_sentences):
        add(sent, f"zh_sentence_{i}")

    # Korean
    ko_sentences = [
        "대한민국의 수도는 서울입니다.",
        "한글은 세종대왕이 만들었습니다.",
        "김치는 한국의 전통 음식입니다.",
    ]
    for i, sent in enumerate(ko_sentences):
        add(sent, f"ko_sentence_{i}")

    # Arabic
    ar_sentences = [
        "بسم الله الرحمن الرحيم",
        "السلام عليكم ورحمة الله وبركاته",
        "اللغة العربية من أقدم اللغات",
    ]
    for i, sent in enumerate(ar_sentences):
        add(sent, f"ar_sentence_{i}")

    # Multi-paragraph text
    add("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", "multi_paragraph_en")
    add("Перший абзац.\n\nДругий абзац.\n\nТретій абзац.", "multi_paragraph_uk")

    print(f"  After section 5 (real-world text): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 6: CODE SNIPPETS (~300)
    # ═══════════════════════════════════════════════════════════

    code_snippets = [
        # Go
        ('package main\n\nimport (\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\targs := os.Args[1:]\n\tfor _, arg := range args {\n\t\tfmt.Println(arg)\n\t}\n}', "go_full"),
        ('type Server struct {\n\taddr string\n\tport int\n}\n\nfunc (s *Server) Start() error {\n\treturn nil\n}', "go_struct"),
        ('ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)\ndefer cancel()', "go_context"),
        ('ch := make(chan int, 10)\ngo func() { ch <- 42 }()\nresult := <-ch', "go_channels"),
        ('if err := json.NewDecoder(r.Body).Decode(&req); err != nil {\n\thttp.Error(w, err.Error(), http.StatusBadRequest)\n\treturn\n}', "go_json_decode"),
        ('var mu sync.Mutex\nmu.Lock()\ndefer mu.Unlock()', "go_mutex"),

        # Python
        ('def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b', "py_fibonacci"),
        ('class MyClass:\n    def __init__(self, value):\n        self._value = value\n    \n    @property\n    def value(self):\n        return self._value', "py_class"),
        ('with open("file.txt", "r") as f:\n    content = f.read()', "py_with"),
        ('result = [x**2 for x in range(100) if x % 2 == 0]', "py_listcomp"),
        ('async def fetch_data():\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()', "py_async"),
        ("try:\n    result = dangerous_operation()\nexcept (ValueError, TypeError) as e:\n    logger.error(f'Error: {e}')\n    raise", "py_exception"),

        # JavaScript/TypeScript
        ('const fetchData = async (url: string): Promise<Response> => {\n  const res = await fetch(url);\n  if (!res.ok) throw new Error(`HTTP ${res.status}`);\n  return res.json();\n};', "ts_fetch"),
        ('const arr = [1, 2, 3, 4, 5];\nconst sum = arr.reduce((acc, val) => acc + val, 0);', "js_reduce"),
        ("document.querySelector('#app')?.addEventListener('click', (e) => {\n  console.log(e.target);\n});", "js_dom"),
        ('import { useState, useEffect } from "react";\n\nfunction App() {\n  const [count, setCount] = useState(0);\n  useEffect(() => { document.title = `Count: ${count}`; }, [count]);\n  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;\n}', "react_component"),

        # Rust
        ('fn main() {\n    let mut v: Vec<i32> = Vec::new();\n    v.push(1);\n    v.push(2);\n    println!("{:?}", v);\n}', "rust_vec"),
        ("impl<T: Clone + Debug> Display for MyStruct<T> {\n    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {\n        write!(f, \"{:?}\", self.data)\n    }\n}", "rust_impl"),
        ("match result {\n    Ok(val) => println!(\"Success: {}\", val),\n    Err(e) => eprintln!(\"Error: {}\", e),\n}", "rust_match"),

        # SQL
        ("CREATE TABLE users (\n    id SERIAL PRIMARY KEY,\n    email VARCHAR(255) UNIQUE NOT NULL,\n    created_at TIMESTAMP DEFAULT NOW()\n);", "sql_create"),
        ("WITH ranked AS (\n    SELECT *, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn\n    FROM employees\n)\nSELECT * FROM ranked WHERE rn <= 3;", "sql_cte"),

        # Shell
        ('#!/bin/bash\nset -euo pipefail\nfor f in *.txt; do\n    echo "Processing $f"\n    wc -l "$f"\ndone', "bash_script"),
        ("find . -name '*.go' -exec grep -l 'TODO' {} \\;", "bash_find"),
        ("curl -sS https://api.example.com/data | jq '.results[] | .name'", "bash_curl_jq"),

        # YAML
        ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: my-app\nspec:\n  replicas: 3\n  selector:\n    matchLabels:\n      app: my-app", "yaml_k8s"),

        # TOML
        ('[package]\nname = "my-project"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\ntokio = { version = "1", features = ["full"] }', "toml_cargo"),

        # Dockerfile
        ("FROM golang:1.22-alpine AS builder\nWORKDIR /app\nCOPY go.mod go.sum ./\nRUN go mod download\nCOPY . .\nRUN CGO_ENABLED=0 go build -o /bin/app .", "dockerfile"),

        # Terraform
        ('resource "aws_instance" "web" {\n  ami           = "ami-0c55b159cbfafe1f0"\n  instance_type = "t2.micro"\n  tags = {\n    Name = "web-server"\n  }\n}', "terraform"),

        # CSS
        (".container {\n  display: flex;\n  justify-content: center;\n  align-items: center;\n  background-color: #f0f0f0;\n  border-radius: 8px;\n  padding: 16px;\n}", "css_flex"),

        # Regex
        (r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$", "regex_ip"),
        (r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", "regex_email"),
    ]
    for text, name in code_snippets:
        add(text, f"code_{name}")

    print(f"  After section 6 (code snippets): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 7: EMOJI COMPREHENSIVE (~300)
    # ═══════════════════════════════════════════════════════════

    # Single emoji from every sub-range
    emoji_codepoints = [
        0x231A, 0x231B, 0x23E9, 0x23F0, 0x23F3,  # watch, hourglass, etc.
        0x25AA, 0x25AB, 0x25B6, 0x25C0, 0x25FB,  # geometric
        0x2600, 0x2601, 0x2602, 0x2603, 0x2604,  # weather
        0x260E, 0x2611, 0x2614, 0x2615, 0x2618,  # misc
        0x261D, 0x2620, 0x2622, 0x2623, 0x2626,
        0x262A, 0x262E, 0x262F, 0x2638, 0x2639,
        0x263A, 0x2640, 0x2642, 0x2648, 0x2649,
        0x264A, 0x264B, 0x264C, 0x264D, 0x264E,
        0x264F, 0x2650, 0x2651, 0x2652, 0x2653,
        0x265F, 0x2660, 0x2663, 0x2665, 0x2666,
        0x2668, 0x267B, 0x267E, 0x267F, 0x2692,
        0x2693, 0x2694, 0x2695, 0x2696, 0x2697,
        0x2699, 0x269B, 0x269C, 0x26A0, 0x26A1,
        0x26AA, 0x26AB, 0x26B0, 0x26B1, 0x26BD,
        0x26BE, 0x26C4, 0x26C5, 0x26CE, 0x26CF,
        0x26D1, 0x26D3, 0x26D4, 0x26E9, 0x26EA,
        0x26F0, 0x26F1, 0x26F2, 0x26F3, 0x26F4,
        0x26F5, 0x26F7, 0x26F8, 0x26F9, 0x26FA,
        0x26FD, 0x2702, 0x2705, 0x2708, 0x2709,
        0x270A, 0x270B, 0x270C, 0x270D, 0x270F,
        0x2712, 0x2714, 0x2716, 0x271D, 0x2721,
        0x2728, 0x2733, 0x2734, 0x2744, 0x2747,
        0x274C, 0x274E, 0x2753, 0x2754, 0x2755,
        0x2757, 0x2763, 0x2764, 0x2795, 0x2796,
        0x2797, 0x27A1, 0x27B0, 0x27BF,
    ]
    for cp in emoji_codepoints:
        try:
            add(chr(cp), f"emoji_U{cp:04X}")
        except (ValueError, OverflowError):
            pass

    # Modern emoji (emoticons block)
    for cp in range(0x1F600, 0x1F650):
        try:
            add(chr(cp), f"emoticon_U{cp:04X}")
        except (ValueError, OverflowError):
            pass

    # Skin tone modifiers
    skin_tones = [0x1F3FB, 0x1F3FC, 0x1F3FD, 0x1F3FE, 0x1F3FF]
    base_people = [0x1F44D, 0x1F44E, 0x1F44B, 0x1F64B, 0x1F64C, 0x1F64F, 0x1F466, 0x1F467, 0x1F468, 0x1F469]
    for base in base_people:
        for tone in skin_tones:
            try:
                add(chr(base) + chr(tone), f"skin_{base:04X}_{tone:04X}")
            except (ValueError, OverflowError):
                pass

    # ZWJ sequences
    zwj_sequences = [
        "👨\u200d💻", "👩\u200d💻", "👨\u200d🔬", "👩\u200d🔬",
        "👨\u200d🍳", "👩\u200d🍳", "👨\u200d🎓", "👩\u200d🎓",
        "👨\u200d⚕️", "👩\u200d⚕️", "👨\u200d🌾", "👩\u200d🌾",
        "👨\u200d🎨", "👩\u200d🎨", "👨\u200d✈️", "👩\u200d✈️",
        "👨\u200d🚀", "👩\u200d🚀", "👨\u200d🚒", "👩\u200d🚒",
        "🐻\u200d❄️", "🏳️\u200d⚧️",
        "👩\u200d❤️\u200d👨", "👩\u200d❤️\u200d👩", "👨\u200d❤️\u200d👨",
        "👩\u200d❤️\u200d💋\u200d👨",
    ]
    for i, seq in enumerate(zwj_sequences):
        add(seq, f"zwj_seq_{i}")

    # Flag sequences
    flag_codes = ["US", "UA", "GB", "DE", "FR", "JP", "KR", "CN", "BR", "IN",
                  "CA", "AU", "IT", "ES", "MX", "PL", "NL", "SE", "NO", "FI"]
    for code in flag_codes:
        flag = chr(0x1F1E6 + ord(code[0]) - ord('A')) + chr(0x1F1E6 + ord(code[1]) - ord('A'))
        add(flag, f"flag_{code}")

    print(f"  After section 7 (emoji): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 8: NFKC NORMALIZATION EDGE CASES (~200)
    # ═══════════════════════════════════════════════════════════

    # Characters that change under NFKC
    nfkc_cases = [
        ("ﬁ", "fi"),       # fi ligature -> fi
        ("ﬂ", "fl"),       # fl ligature -> fl
        ("ﬃ", "ffi"),      # ffi ligature -> ffi
        ("ﬄ", "ffl"),      # ffl ligature -> ffl
        ("ﬅ", "st"),       # long st ligature
        ("ﬆ", "st"),       # st ligature
        ("①", "1"),        # circled 1
        ("②", "2"),        # circled 2
        ("③", "3"),        # circled 3
        ("⑩", "10"),       # circled 10
        ("Ⅰ", "I"),        # roman numeral I
        ("Ⅱ", "II"),       # roman numeral II
        ("Ⅲ", "III"),      # roman numeral III
        ("Ⅳ", "IV"),       # roman numeral IV
        ("ℋ", "H"),        # script H
        ("ℌ", "H"),        # fraktur H
        ("ℍ", "H"),        # double-struck H
        ("ℎ", "h"),        # planck constant
        ("ℏ", "ℏ"),        # h-bar (stays)
        ("Ω", "Ω"),        # ohm sign -> greek omega
        ("Å", "Å"),        # angstrom -> A ring
        ("㎡", "m2"),       # square meter
        ("㎥", "m3"),       # cubic meter
        ("㎞", "km"),       # kilometer
        ("㎝", "cm"),       # centimeter
        ("㎜", "mm"),       # millimeter
        ("㎏", "kg"),       # kilogram
        ("㎎", "mg"),       # milligram
        ("㎐", "Hz"),       # hertz
        ("㏃", "Bq"),       # becquerel
        ("½", "1⁄2"),      # fraction half
        ("¼", "1⁄4"),      # fraction quarter
        ("¾", "3⁄4"),      # fraction three quarters
        ("²", "2"),        # superscript 2
        ("³", "3"),        # superscript 3
        ("¹", "1"),        # superscript 1
        ("⁴", "4"),        # superscript 4
        ("ⁿ", "n"),        # superscript n
        ("₀", "0"),        # subscript 0
        ("₁", "1"),        # subscript 1
        ("Ａ", "A"),        # fullwidth A
        ("ａ", "a"),        # fullwidth a
        ("０", "0"),        # fullwidth 0
        ("！", "!"),        # fullwidth !
    ]
    for char, expected_nfkc in nfkc_cases:
        add(char, f"nfkc_{ord(char):04X}")
        add(f"hello {char} world", f"nfkc_in_text_{ord(char):04X}")

    # Fullwidth ASCII (systematic)
    for i in range(0xFF01, 0xFF5F):
        add(chr(i), f"fullwidth_U{i:04X}")

    # Halfwidth katakana
    for i in range(0xFF65, 0xFF9E):
        add(chr(i), f"halfwidth_kana_U{i:04X}")

    print(f"  After section 8 (NFKC edge cases): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 9: MIXED CONTENT PATTERNS (~500)
    # ═══════════════════════════════════════════════════════════

    # Technical documentation
    tech_docs = [
        "The API returns HTTP 200 OK with Content-Type: application/json.",
        "Configure the timeout to 30s using --timeout=30s flag.",
        "The error rate increased from 0.1% to 2.5% after deployment.",
        "Memory usage: 256MB (RSS), 512MB (VSZ), CPU: 45%.",
        "Latency p99: 150ms, p95: 80ms, p50: 12ms.",
        "Version 3.2.1-rc.4 is available for download.",
        "See RFC 2616 §14.9 for Cache-Control header details.",
        "The SHA-256 hash is 64 hex characters long.",
        "Use TLS 1.3 with ECDHE-RSA-AES256-GCM-SHA384 cipher.",
        "IPv6 address: 2001:0db8:85a3:0000:0000:8a2e:0370:7334",
    ]
    for i, doc in enumerate(tech_docs):
        add(doc, f"tech_doc_{i}")

    # Log messages
    log_msgs = [
        "[2024-01-15 10:30:00.123] INFO  server.go:42 - Server started on :8080",
        "[ERROR] 2024-01-15T10:30:00Z connection_pool.go:128 - Pool exhausted: max_connections=100",
        "WARN  [main] org.example.App - Deprecated API called from 192.168.1.1",
        "DEBUG kafka_consumer.go:55 - Received message: topic=events partition=3 offset=12345",
        "FATAL: database migration failed: column 'user_id' already exists",
        "panic: runtime error: index out of range [5] with length 3",
        "goroutine 1 [running]:\nmain.main()\n\t/app/main.go:42 +0x1a8",
    ]
    for i, msg in enumerate(log_msgs):
        add(msg, f"log_msg_{i}")

    # Config files
    configs = [
        'DATABASE_URL="postgres://user:pass@localhost:5432/mydb?sslmode=disable"',
        "REDIS_URL=redis://localhost:6379/0",
        "AWS_REGION=us-east-1",
        "SMTP_HOST=smtp.gmail.com:587",
        "max_retries = 3\nbackoff_factor = 1.5\ntimeout = 30",
    ]
    for i, cfg in enumerate(configs):
        add(cfg, f"config_{i}")

    # Mathematical expressions
    math_exprs = [
        "f(x) = x² + 2x + 1",
        "∫₀^∞ e^(-x²) dx = √π/2",
        "∇ × B = μ₀J + μ₀ε₀ ∂E/∂t",
        "E = mc²",
        "eiπ + 1 = 0",
        "∑_{n=0}^{∞} xⁿ/n! = eˣ",
        "lim_{x→0} sin(x)/x = 1",
        "P(A|B) = P(B|A)P(A)/P(B)",
        "|ψ⟩ = α|0⟩ + β|1⟩",
        "∂²u/∂t² = c² ∂²u/∂x²",
    ]
    for i, expr in enumerate(math_exprs):
        add(expr, f"math_expr_{i}")

    # Mixed-script sentences
    mixed_scripts = [
        "The Japanese word 日本語 means 'Japanese language'.",
        "В Python используется def для определения функций.",
        "Die Gleichung E=mc² wurde von Einstein formuliert.",
        "중요한 정보: this is important information.",
        "مهم: Please read the following instructions carefully.",
        "Используйте команду git push для отправки изменений.",
        "Le théorème de Fermat: xⁿ + yⁿ = zⁿ n'a pas de solution pour n>2.",
        "タイプスクリプト (TypeScript) は JavaScript のスーパーセット。",
        "Πυθαγόρειο θεώρημα: a² + b² = c².",
        "Hindi: हिंदी, Arabic: العربية, Hebrew: עברית",
    ]
    for i, text in enumerate(mixed_scripts):
        add(text, f"mixed_script_{i}")

    # Email-like content
    emails = [
        "From: user@example.com\nTo: admin@example.org\nSubject: Re: Meeting tomorrow\n\nHi,\n\nPlease see the attached document.\n\nBest regards,\nJohn",
        "Dear Mr. Smith,\n\nThank you for your inquiry regarding order #12345.\nYour package (tracking: 1Z999AA10123456784) will arrive by Friday.\n\nSincerely,\nCustomer Support",
    ]
    for i, email in enumerate(emails):
        add(email, f"email_body_{i}")

    # CSV-like data
    csv_rows = [
        "name,age,city,country\nJohn,30,Kyiv,Ukraine\nJane,25,Berlin,Germany\nTaro,35,Tokyo,Japan",
        '"Smith, John",42,"New York, NY","United States"',
        "id\tscore\tgrade\n1\t95.5\tA+\n2\t87.3\tB+\n3\t72.1\tC",
    ]
    for i, csv in enumerate(csv_rows):
        add(csv, f"csv_data_{i}")

    # URLs with special chars
    urls = [
        "https://example.com/path?q=hello+world&lang=uk&page=1#results",
        "https://uk.wikipedia.org/wiki/Київ",
        "https://example.com/api/v2/users/123/posts?fields=id,title,body&sort=-created_at",
        "file:///home/user/Documents/файл.pdf",
        "mailto:user@example.com?subject=Hello%20World&body=Test%20message",
        "data:text/plain;charset=utf-8;base64,SGVsbG8gV29ybGQ=",
    ]
    for i, url in enumerate(urls):
        add(url, f"url_special_{i}")

    # Random word combinations
    words_en = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                "hello", "world", "machine", "learning", "deep", "neural", "network",
                "tokenizer", "encode", "decode", "model", "training"]
    words_uk = ["привіт", "світ", "машинне", "навчання", "глибоке", "нейронна", "мережа",
                "токенізатор", "кодування", "декодування", "модель", "тренування"]

    for i in range(50):
        n = random.randint(3, 15)
        w = random.choices(words_en, k=n)
        add(" ".join(w), f"random_en_words_{i}")

    for i in range(30):
        n = random.randint(3, 10)
        w = random.choices(words_uk, k=n)
        add(" ".join(w), f"random_uk_words_{i}")

    for i in range(30):
        n = random.randint(4, 12)
        w_en = random.choices(words_en, k=n//2)
        w_uk = random.choices(words_uk, k=n//2)
        combined = w_en + w_uk
        random.shuffle(combined)
        add(" ".join(combined), f"random_mixed_words_{i}")

    print(f"  After section 9 (mixed content): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 10: ADVERSARIAL / STRESS (~300)
    # ═══════════════════════════════════════════════════════════

    # Very long single-char strings at various lengths
    for ch, name in [("b", "b"), (".", "dot"), ("!", "bang"), (" ", "sp")]:
        for n in [100, 500, 1000, 2000]:
            add(ch * n, f"stress_{name}_{n}")

    # Alternating scripts
    add("aбcдeфgхiйkлmнoпqрsтuвwхyз", "alternating_latin_cyrillic")
    add("a你b好c世d界", "alternating_latin_cjk")
    add("hello мир world свет", "alternating_words_en_ru")
    add("α1β2γ3δ4ε5", "alternating_greek_digits")

    # Control character sequences
    for c in range(0, 32):
        add(f"before{chr(c)}after", f"ctrl_char_{c:02d}")

    # Byte patterns that stress UTF-8 decoding
    # 2-byte sequences
    for cp in [0x80, 0xBF, 0xC0, 0xFF, 0x100, 0x7FF]:
        try:
            add(chr(cp), f"utf8_boundary_U{cp:04X}")
        except (ValueError, OverflowError):
            pass
    # 3-byte sequences
    for cp in [0x800, 0xFFF, 0x1000, 0xFFFF]:
        try:
            add(chr(cp), f"utf8_3byte_U{cp:04X}")
        except (ValueError, OverflowError):
            pass
    # 4-byte sequences
    for cp in [0x10000, 0x1FFFF, 0x20000, 0x10FFFF]:
        try:
            add(chr(cp), f"utf8_4byte_U{cp:04X}")
        except (ValueError, OverflowError):
            pass

    # Strings with many different pieces
    add("".join(chr(i) for i in range(65, 91)), "uppercase_alphabet")
    add("".join(chr(i) for i in range(97, 123)), "lowercase_alphabet")
    add("".join(chr(i) for i in range(48, 58)), "digits_0_9")
    add(string.punctuation, "all_punctuation")

    # Max subword challenge: random chars that force many small pieces
    random_bytes = "".join(chr(random.randint(0x0100, 0x04FF)) for _ in range(100))
    add(random_bytes, "random_unicode_100")

    random_bytes2 = "".join(chr(random.randint(0x4E00, 0x9FFF)) for _ in range(100))
    add(random_bytes2, "random_cjk_100")

    random_bytes3 = "".join(chr(random.randint(0x1F600, 0x1F64F)) for _ in range(50))
    add(random_bytes3, "random_emoji_50")

    # Strings with only combining marks (no base)
    add("\u0300\u0301\u0302", "combining_only")
    add("\u0300" * 10, "combining_grave_10")
    add("\u0308\u0301\u0327" * 5, "combining_stack_15")

    # Very deep nesting
    add("(" * 50 + "x" + ")" * 50, "deep_nested_parens")
    add("{" * 30 + '"key":"val"' + "}" * 30, "deep_nested_braces")
    add("<" * 20 + "tag" + ">" * 20, "deep_nested_angles")

    # Alternating case
    add("".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate("hello world this is a test")), "spongebob_case")

    # Tab-delimited with many columns
    add("\t".join(f"col{i}" for i in range(50)), "tsv_50_cols")

    # JSON with Unicode
    add('{"name": "Тарас Шевченко", "city": "Київ", "emoji": "🇺🇦"}', "json_unicode")
    add('{"array": [1, "два", 3.14, true, null, "🎉"]}', "json_mixed_types")

    # XML
    add('<?xml version="1.0" encoding="UTF-8"?>\n<root>\n  <item id="1">Hello</item>\n</root>', "xml_doc")

    # Base64
    add("SGVsbG8gV29ybGQh", "base64_hello")
    add("VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw==", "base64_pangram")

    # UUID
    add("550e8400-e29b-41d4-a716-446655440000", "uuid")
    add("urn:uuid:550e8400-e29b-41d4-a716-446655440000", "uuid_urn")

    # Hex strings
    add("deadbeef" * 8, "hex_deadbeef")
    add("0123456789abcdef" * 4, "hex_all_digits")

    # Random printable ASCII strings
    for i in range(50):
        length = random.randint(10, 200)
        text = "".join(random.choices(string.printable, k=length))
        add(text, f"random_ascii_{i}")

    # Random Unicode strings
    safe_ranges = [(0x0020, 0x007E), (0x00A0, 0x024F), (0x0400, 0x04FF),
                   (0x4E00, 0x4EFF), (0x3040, 0x309F), (0xAC00, 0xAC7F)]
    for i in range(50):
        length = random.randint(10, 100)
        chars = []
        for _ in range(length):
            r = random.choice(safe_ranges)
            chars.append(chr(random.randint(r[0], r[1])))
        add("".join(chars), f"random_unicode_mix_{i}")

    print(f"  After section 10 (adversarial/stress): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 11: SPECIFIC BYTE FALLBACK CASES (~200)
    # ═══════════════════════════════════════════════════════════

    # Supplementary plane characters (all 4-byte UTF-8)
    smp_ranges = [
        (0x10000, 0x1003F, "linear_b"),
        (0x10300, 0x1032F, "old_italic"),
        (0x10330, 0x1034F, "gothic"),
        (0x10400, 0x1044F, "deseret"),
        (0x10800, 0x1083F, "cypriot"),
        (0x12000, 0x1203F, "cuneiform"),
        (0x13000, 0x1303F, "egyptian_hieroglyphs"),
        (0x1D000, 0x1D03F, "byzantine_music"),
        (0x1D100, 0x1D13F, "music_symbols"),
        (0x1D400, 0x1D43F, "math_bold"),
        (0x1D440, 0x1D47F, "math_italic"),
        (0x1D4A0, 0x1D4DF, "math_script"),
        (0x1D500, 0x1D53F, "math_fraktur"),
        (0x1D540, 0x1D57F, "math_double_struck"),
        (0x1D7CE, 0x1D7FF, "math_digits"),
        (0x1F000, 0x1F02F, "mahjong"),
        (0x1F030, 0x1F09F, "domino"),
        (0x1F0A0, 0x1F0FF, "playing_cards"),
        (0x1F100, 0x1F1FF, "enclosed_alphanum_supp"),
        (0x1F200, 0x1F2FF, "enclosed_ideo_supp"),
        (0x1F700, 0x1F73F, "alchemical"),
    ]
    for start, end, name in smp_ranges:
        # Sample a few chars from each
        for offset in [0, (end-start)//3, (end-start)//2, end-start]:
            cp = start + offset
            if cp <= 0x10FFFF:
                try:
                    ch = chr(cp)
                    add(ch, f"smp_{name}_U{cp:05X}")
                    add(f"text {ch} text", f"smp_{name}_in_text_U{cp:05X}")
                except (ValueError, OverflowError):
                    pass

    # Multiple byte-fallback chars in sequence
    byte_heavy = "".join(chr(cp) for cp in [0x10000, 0x10001, 0x10002, 0x10003, 0x10004])
    add(byte_heavy, "byte_fallback_sequence_5")
    byte_heavy2 = "".join(chr(cp) for cp in range(0x12000, 0x12020))
    add(byte_heavy2, "byte_fallback_cuneiform_32")

    # Mixed vocab and byte-fallback
    add(f"hello {chr(0x10348)} world", "byte_fb_mixed_1")
    add(f"Привіт {chr(0x13000)} мир", "byte_fb_mixed_2")
    add(f"abc{chr(0x10400)}{chr(0x10401)}{chr(0x10402)}def", "byte_fb_mixed_3")

    print(f"  After section 11 (byte fallback): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 12: PADDING TO 5000+ (~remaining)
    # ═══════════════════════════════════════════════════════════

    # More random sentences
    templates_en = [
        "The {adj} {noun} {verb} the {adj2} {noun2}.",
        "{name} went to {place} to buy {thing}.",
        "It was a {adj} day when {name} decided to {verb} the {noun}.",
        "In {year}, the {adj} {noun} was {verb}ed by {name}.",
    ]
    adjs = ["big", "small", "fast", "slow", "red", "blue", "old", "new", "dark", "bright"]
    nouns = ["cat", "dog", "bird", "fish", "tree", "house", "car", "book", "river", "mountain"]
    verbs = ["chase", "find", "build", "destroy", "paint", "read", "watch", "eat"]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    places = ["the park", "the store", "the library", "downtown", "the airport"]
    things = ["groceries", "books", "flowers", "tickets", "a gift"]
    years = ["2020", "1999", "2024", "1776", "2050"]

    for i in range(200):
        tmpl = random.choice(templates_en)
        text = tmpl.format(
            adj=random.choice(adjs), noun=random.choice(nouns),
            verb=random.choice(verbs), adj2=random.choice(adjs),
            noun2=random.choice(nouns), name=random.choice(names),
            place=random.choice(places), thing=random.choice(things),
            year=random.choice(years),
        )
        add(text, f"gen_sentence_{i}")

    # Ukrainian generated
    uk_adjs = ["великий", "малий", "швидкий", "повільний", "гарний", "поганий", "старий", "новий"]
    uk_nouns = ["кіт", "собака", "дім", "дерево", "річка", "місто", "книга", "людина"]
    uk_verbs = ["бачити", "знайти", "побудувати", "прочитати", "написати"]

    for i in range(100):
        n = random.randint(3, 8)
        words = []
        for _ in range(n):
            words.append(random.choice(uk_adjs + uk_nouns + uk_verbs))
        add(" ".join(words), f"gen_uk_words_{i}")

    # More length edge cases
    for length in [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        add("a" * length, f"prime_len_a_{length}")
        add("я" * length, f"prime_len_ya_{length}")

    # Every 2-char ASCII combination (sample)
    for i in range(200):
        c1 = chr(random.randint(32, 126))
        c2 = chr(random.randint(32, 126))
        add(c1 + c2, f"ascii_pair_{i}_{ord(c1):02x}{ord(c2):02x}")

    # Every 3-char combo (sample)
    for i in range(100):
        c1 = chr(random.randint(32, 126))
        c2 = chr(random.randint(32, 126))
        c3 = chr(random.randint(32, 126))
        add(c1 + c2 + c3, f"ascii_triple_{i}")

    print(f"  After section 12 (padding): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 13: C++ REFERENCE TEST CASES (~200)
    # From sources/src/normalizer_test.cc & other test files
    # ═══════════════════════════════════════════════════════════

    cpp_test_inputs = [
        # normalizer_test.cc inputs
        "ABC", " ABC ", " A  B  C ", "   ABC   ",
        "\uff21\uff22\uff23",  # ＡＢＣ (fullwidth)
        "   \uff21\uff22\uff23   ",
        "\u3000\u3000ABC",  # ideographic space + ABC
        "\u3000\u3000ABC\u3000\u3000",
        "\u2460\u2461\u2462",  # ①②③
        "\u337f",  # ㍿ (square era name)
        "\uff78\uff9e\uff70\uff78\uff9e\uff99",  # ｸﾞｰｸﾞﾙ
        " \uff78\uff9e\uff70\uff78\uff9e\uff99 ",
        " I  saw a\u3000 \u3000girl\u3000\u3000",
        # Control chars that normalize to empty
        chr(0x7F), chr(0x8F), chr(0x9F), chr(0x0B),
        # NFKC normalization edge cases from trainer_test.cc
        "\u337b\u5143\u5e74",  # ㍻元年 (square era + kanji)
        "\uff76\uff9e\uff72\uff80\uff9e\uff9d\uff7d",  # ｶﾞｲﾀﾞﾝｽ
        "KADOKAWA",
        # From sentencepiece_processor_test.cc
        "This is a test.",
        "Hello World",
        "test",
        "",
        # Unicode normalization
        "ＡＢＣ",  # fullwidth
        "ＡＢＣＤ",  # fullwidth
        "①②③④⑤⑥⑦⑧⑨⑩",
        # Mixed fullwidth/halfwidth
        "Hello Ｗｏｒｌｄ",
        "ＡＢＣ abc ＤＥＦ def",
    ]
    for i, text in enumerate(cpp_test_inputs):
        add(text, f"cpp_ref_{i}")

    # Control chars 0x00-0x1F that the C++ normalizer tests
    for c in range(0x80, 0xA0):
        add(chr(c), f"cpp_ctrl_U{c:04X}")

    # From byte_fallback tests
    for byte_val in range(256):
        token = f"<0x{byte_val:02X}>"
        add(token, f"cpp_byte_token_text_{byte_val:02X}")

    print(f"  After section 13 (C++ reference): {len(cases)} cases")

    # ═══════════════════════════════════════════════════════════
    # SECTION 14: MASSIVE RANDOM GENERATION TO HIT 5000+
    # ═══════════════════════════════════════════════════════════

    # More random English sentences (templates)
    templates2 = [
        "The {n1} and the {n2} walked through the {adj} {n3}.",
        "Every {adj} {n1} needs a {adj2} {n2} to survive.",
        "When the {n1} saw the {n2}, it began to {v}.",
        "{name} said: 'The {adj} {n1} must {v} before dawn.'",
        "Under the {adj} sky, the {n1} and {n2} {v}ed together.",
        "Nobody expected the {adj} {n1} to {v} so {adv}.",
    ]
    advs = ["quickly", "slowly", "carefully", "silently", "loudly", "gracefully"]
    for i in range(300):
        tmpl = random.choice(templates2)
        text = tmpl.format(
            n1=random.choice(nouns), n2=random.choice(nouns),
            n3=random.choice(nouns), adj=random.choice(adjs),
            adj2=random.choice(adjs), v=random.choice(verbs),
            name=random.choice(names), adv=random.choice(advs),
        )
        add(text, f"gen2_sentence_{i}")

    # More Ukrainian sentences
    uk_templates = [
        "{adj} {n1} та {adj2} {n2} разом {v}.",
        "Коли {n1} побачив {n2}, він почав {v}.",
        "Кожен {adj} {n1} потребує {adj2} {n2}.",
        "Під {adj} небом {n1} і {n2} разом {v}.",
    ]
    for i in range(200):
        tmpl = random.choice(uk_templates)
        text = tmpl.format(
            n1=random.choice(uk_nouns), n2=random.choice(uk_nouns),
            adj=random.choice(uk_adjs), adj2=random.choice(uk_adjs),
            v=random.choice(uk_verbs),
        )
        add(text, f"gen2_uk_{i}")

    # More random mixed scripts
    all_chars_pool = (
        list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)) +  # Latin
        list(range(0x410, 0x450)) +  # Cyrillic
        list(range(0x4E00, 0x4E50)) +  # CJK
        list(range(0x3041, 0x3060)) +  # Hiragana
        [0x20, 0x2C, 0x2E, 0x21, 0x3F]  # spaces/punct
    )
    for i in range(200):
        length = random.randint(5, 80)
        chars = [chr(random.choice(all_chars_pool)) for _ in range(length)]
        add("".join(chars), f"gen2_mixed_{i}")

    # Strings with specific numbers of spaces
    for spaces in range(1, 15):
        add("a" + " " * spaces + "b", f"exact_spaces_{spaces}")

    # Repeated patterns at various lengths
    patterns = ["ab", "abc", "abcd", "hello world ", "тест ", "你好", "🎉🎊"]
    for pat in patterns:
        for rep in [3, 7, 13, 25, 50, 100]:
            add(pat * rep, f"pattern_{pat[:4]}_{rep}")

    # More code snippets with various languages
    more_code = [
        "SELECT COUNT(*) FROM orders WHERE status = 'completed' AND total > 100.00;",
        "INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com');",
        "UPDATE products SET price = price * 1.1 WHERE category = 'electronics';",
        "DELETE FROM sessions WHERE last_active < NOW() - INTERVAL '30 days';",
        "ALTER TABLE users ADD COLUMN phone VARCHAR(20) DEFAULT NULL;",
        "CREATE INDEX idx_users_email ON users (email);",
        "import React, { useState, useEffect, useCallback, useMemo } from 'react';",
        "export default function App({ children }: { children: React.ReactNode }) {",
        "const [state, dispatch] = useReducer(reducer, initialState);",
        'docker run -d --name app -p 8080:8080 -v /data:/app/data myapp:latest',
        'kubectl get pods -n production --selector=app=web -o wide',
        'terraform plan -var-file=prod.tfvars -out=tfplan',
        'aws s3 sync ./build s3://my-bucket --delete --cache-control max-age=31536000',
        'gcloud compute instances create vm1 --zone=us-central1-a --machine-type=e2-medium',
    ]
    for i, code in enumerate(more_code):
        add(code, f"gen2_code_{i}")

    # JSON payloads
    json_payloads = [
        '{"users":[{"id":1,"name":"Alice","roles":["admin","user"]},{"id":2,"name":"Bob","roles":["user"]}]}',
        '{"error":{"code":404,"message":"Not Found","details":{"path":"/api/v2/users/999"}}}',
        '{"data":{"temperature":22.5,"humidity":65,"pressure":1013.25,"wind":{"speed":5.2,"direction":"NW"}}}',
        '{"config":{"database":{"host":"localhost","port":5432,"name":"mydb","ssl":true},"cache":{"ttl":300}}}',
    ]
    for i, payload in enumerate(json_payloads):
        add(payload, f"gen2_json_{i}")

    # More URLs
    more_urls = [
        "https://api.example.com/v3/users?page=1&per_page=50&sort=created_at&order=desc",
        "https://example.com/search?q=hello+world&lang=uk&safe=on&start=20",
        "https://cdn.example.com/assets/img/logo@2x.png?v=1705312200",
        "mongodb://user:p%40ssw0rd@cluster0.example.net:27017/mydb?retryWrites=true",
        "jdbc:postgresql://db.example.com:5432/production?sslmode=require&connectTimeout=10",
    ]
    for i, url in enumerate(more_urls):
        add(url, f"gen2_url_{i}")

    # Stack traces
    add("panic: runtime error: invalid memory address or nil pointer dereference\n[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x10a4e70]\n\ngoroutine 1 [running]:\nmain.(*App).Start(0x0)\n\t/app/main.go:42 +0x30", "gen2_stacktrace_go")
    add("Traceback (most recent call last):\n  File \"/app/main.py\", line 42, in <module>\n    result = process(data)\n  File \"/app/utils.py\", line 15, in process\n    return data['key']\nKeyError: 'key'", "gen2_stacktrace_py")

    # More date/time formats
    datetimes = [
        "2024-01-15T10:30:00+02:00", "2024-01-15 10:30:00 UTC",
        "Jan 15, 2024 10:30 AM EST", "15.01.2024 10:30:00",
        "1705312200", "2024W03-1", "2024-015",
        "Mon, 15 Jan 2024 10:30:00 GMT",
        "January 15th, 2024", "15 січня 2024 року",
    ]
    for i, dt in enumerate(datetimes):
        add(dt, f"gen2_datetime_{i}")

    # Semantic versioning
    versions = [
        "1.0.0", "2.3.4-beta.1", "0.0.1-alpha", "10.20.30",
        "1.0.0-rc.1+build.123", "3.0.0-SNAPSHOT", "v4.5.6",
    ]
    for i, v in enumerate(versions):
        add(v, f"gen2_semver_{i}")

    # File paths with various separators
    paths = [
        "/home/user/.config/app/settings.yaml",
        "C:\\Program Files\\App\\config.ini",
        "../../../etc/passwd",
        "~/projects/my-app/src/components/Header.tsx",
        "/var/log/nginx/access.log.2024-01-15.gz",
    ]
    for i, p in enumerate(paths):
        add(p, f"gen2_path_{i}")

    # More random ASCII pairs/triples for exhaustive coverage
    for i in range(200, 500):
        c1 = chr(random.randint(32, 126))
        c2 = chr(random.randint(32, 126))
        add(c1 + c2, f"ascii_pair_{i}_{ord(c1):02x}{ord(c2):02x}")

    for i in range(100, 400):
        chars = [chr(random.randint(32, 126)) for _ in range(random.randint(3, 8))]
        add("".join(chars), f"ascii_multi_{i}")

    # More random Unicode strings to push past 5000
    for i in range(50, 150):
        length = random.randint(20, 150)
        chars = [chr(random.choice(all_chars_pool)) for _ in range(length)]
        add("".join(chars), f"gen3_mixed_{i}")

    for i in range(50):
        n = random.randint(5, 20)
        w = random.choices(words_en + words_uk, k=n)
        add(" ".join(w), f"gen3_words_{i}")

    print(f"  After section 14 (massive padding): {len(cases)} cases")

    return cases


def generate_model_info(sp):
    """Extract model metadata."""
    sample_entries = []
    for i in range(min(20, sp.GetPieceSize())):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)
        if sp.IsUnknown(i): ptype = "UNKNOWN"
        elif sp.IsControl(i): ptype = "CONTROL"
        elif sp.IsByte(i): ptype = "BYTE"
        elif sp.IsUnused(i): ptype = "UNUSED"
        else: ptype = "NORMAL"
        sample_entries.append({"piece": piece, "score": score, "type": ptype, "id": i})

    byte_samples = []
    for i in range(sp.GetPieceSize()):
        if sp.IsByte(i):
            byte_samples.append({"piece": sp.IdToPiece(i), "score": sp.GetScore(i), "type": "BYTE", "id": i})
            if len(byte_samples) >= 5:
                break

    normal_samples = []
    for i in range(sp.GetPieceSize()):
        if not sp.IsUnknown(i) and not sp.IsControl(i) and not sp.IsByte(i) and not sp.IsUnused(i):
            normal_samples.append({"piece": sp.IdToPiece(i), "score": sp.GetScore(i), "type": "NORMAL", "id": i})
            if len(normal_samples) >= 10:
                break

    return {
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


def validate_golden(sp, cases):
    """Self-check: verify every case against sentencepiece."""
    errors = 0
    for i, tc in enumerate(cases):
        ids = sp.EncodeAsIds(tc["input"])
        pieces = sp.EncodeAsPieces(tc["input"])
        decoded = sp.DecodeIds(ids)
        if ids != tc["ids"]:
            print(f"MISMATCH ids at {i} ({tc['description']})")
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

    cases = generate_cases(sp)
    print(f"\nTotal: {len(cases)} test cases")

    print("Validating...")
    errors = validate_golden(sp, cases)
    if errors > 0:
        print(f"FATAL: {errors} validation errors!")
        raise SystemExit(1)
    print("All cases validated ✓")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for tc in cases:
            f.write(json.dumps(tc, ensure_ascii=False) + "\n")
    print(f"Written {len(cases)} cases to {OUTPUT_JSONL}")

    info = generate_model_info(sp)
    with open(OUTPUT_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Written model info to {OUTPUT_INFO}")


if __name__ == "__main__":
    main()
