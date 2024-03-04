"""The original repository has a lot of dependencies and
import/version issues that are not necessary for a simple text-to-speech
use case, so I trimmed the requirements to the bare minimum to avoid
potential (future) errors and to make the code easier to install.
"""

# Standard library
import torch
import time
import tempfile
import re
import pathlib
from typing import List
from typing import Dict, Any

# External libraries
from pydub import AudioSegment
import scipy
from unidecode import unidecode
import inflect
from dp.phonemizer import Phonemizer

# SYMBOLS ===========================================================================================
# Phonemes
_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧ"
_diacrilics = "ɚ˞ɫ"
_extra_phons = [
    "g",
    "ɝ",
    "̃",
    "̍",
    "̥",
    "̩",
    "̯",
    "͡",
]  # some extra symbols that I found in from wiktionary ipa annotations
phonemes = (
    list(
        _pad
        + _punctuation
        + _special
        + _vowels
        + _non_pulmonic_consonants
        + _pulmonic_consonants
        + _suprasegmentals
        + _other_symbols
        + _diacrilics
    )
    + _extra_phons
)
phonemes_set = set(phonemes)
# END SYMBOLS =======================================================================================


# RAW IMPORTS =======================================================================================
class Tokenizer:
    def __init__(self) -> None:
        self.symbol_to_id = {s: i for i, s in enumerate(phonemes)}
        self.id_to_symbol = {i: s for i, s in enumerate(phonemes)}

    def __call__(self, text: str) -> List[int]:
        return [self.symbol_to_id[t] for t in text if t in self.symbol_to_id]

    def decode(self, sequence: List[int]) -> str:
        text = [self.id_to_symbol[s] for s in sequence if s in self.id_to_symbol]
        return "".join(text)


""" from https://github.com/keithito/tacotron """
_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def no_cleaners(text):
    return text


def english_cleaners(text):
    text = unidecode(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    return text


class Cleaner:
    def __init__(
        self, cleaner_name: str, use_phonemes: bool, lang: str, model_dir: pathlib.Path
    ) -> None:
        if cleaner_name == "english_cleaners":
            self.clean_func = english_cleaners
        elif cleaner_name == "no_cleaners":
            self.clean_func = no_cleaners
        else:
            raise ValueError(
                f"Cleaner not supported: {cleaner_name}! "
                f"Currently supported: ['english_cleaners', 'no_cleaners']"
            )
        self.use_phonemes = use_phonemes
        self.lang = lang
        if use_phonemes:
            self.phonemize = Phonemizer.from_checkpoint(
                model_dir / "en_us_cmudict_ipa_forward.pt"
            )

    def __call__(self, text: str) -> str:
        text = self.clean_func(text)
        if self.use_phonemes:
            text = self.phonemize(text, lang="en_us")
            text = "".join([p for p in text if p in phonemes_set])
        text = collapse_whitespace(text)
        text = text.strip()
        return text

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Cleaner":
        return Cleaner(
            cleaner_name=config["preprocessing"]["cleaner_name"],
            use_phonemes=config["preprocessing"]["use_phonemes"],
            lang=config["preprocessing"]["language"],
        )


# END RAW IMPORTS ===================================================================================


def prepare_text(text: str, model_dir: pathlib.Path) -> str:
    if not ((text[-1] == ".") or (text[-1] == "?") or (text[-1] == "!")):
        text = text + "."
    cleaner = Cleaner("english_cleaners", True, "en-us", model_dir=model_dir)
    tokenizer = Tokenizer()
    return torch.as_tensor(
        tokenizer(cleaner(text)), dtype=torch.long, device="cpu"
    ).unsqueeze(0)


# MODELS
def get_all(model_dir: pathlib.Path, device: str = "cpu"):
    emb = torch.load(model_dir / "glados_p2.pt")
    glados = torch.jit.load(model_dir / "glados-new.pt")
    vocoder = torch.jit.load(model_dir / "vocoder-gpu.pt", map_location=device)
    for i in range(2):
        init = glados.generate_jit(prepare_text(str(i), model_dir=model_dir), emb, 1.0)
        init_mel = init["mel_post"].to(device)
        init_vo = vocoder(init_mel)

    return emb, glados, vocoder, device
