from typing import List
import tokenizers
from tokenizers import Encoding
from tokenizers.pre_tokenizers import Split, Sequence, UnicodeScripts
from tokenizers.models import WordLevel
from tokenizers.decoders import Replace, Sequence as DeSequence
from tokenizers.processors import TemplateProcessing, PostProcessor

from transformers import PreTrainedTokenizerFast


class UciTokenizer(PreTrainedTokenizerFast):
    itos = {
        0: " ",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: ";",
        10: "#",
        11: "a",
        12: "b",
        13: "c",
        14: "d",
        15: "e",
        16: "f",
        17: "g",
        18: "h",
        19: "n",
        20: "r",
        21: "q",
        22: "k",
    }
    """Integer to String mapping"""

    stoi = {
        " ": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        ";": 9,
        "#": 10,
        "a": 11,
        "b": 12,
        "c": 13,
        "d": 14,
        "e": 15,
        "f": 16,
        "g": 17,
        "h": 18,
        "n": 19,
        "r": 20,
        "q": 21,
        "k": 22,
    }
    """String to Integer Mapping. This is the vocab"""

    def __init__(self):
        tok_model = WordLevel(vocab=self.stoi, unk_token=" ")

        # Pre-tokenizer splits input into single characters
        pre_tokenizer = Split(pattern=tokenizers.Regex(r"."), behavior="isolated")

            
            # Sequence( # uncomment to treat ; as regular token
            # [
            #     Split(pattern=tokenizers.Regex("^;+"), behavior="removed"),
            #     Split(pattern=tokenizers.Regex(r"."), behavior="isolated"),
            # ])
            
        # post processing adds bos unless explicitly ignored by 'add_special_tokens=False'
        post_proc = TemplateProcessing(
            single="; $0",
            pair=None,
            special_tokens=[
                (";", 9),
                ("", 0),
                # this empty token ("", 0) is required. If it's not present, an error is raised
            ],
        )

        # post_proc = ReplaceDoubleBosProcessor()

        slow_tokenizer = tokenizers.Tokenizer(tok_model)
        slow_tokenizer.add_special_tokens([";"])  # bos token
        slow_tokenizer.pre_tokenizer = pre_tokenizer
        slow_tokenizer.post_processor = post_proc

        # Now that the tokenizer is built, wrap it so its compatible with HookedTransformer
        super().__init__(
            tokenizer_object=slow_tokenizer,
            bos_token=";",
            name_or_path=None,
            add_bos_token=True,
        )

        # Cleans up added spaces
        meta_symbol = "▁"  # not an underscore! (U+2581)
        self._tokenizer.decoder = DeSequence(
            [Replace(" ", meta_symbol), Replace(meta_symbol, " ")]
        )


if __name__ == "__main__":
    fast_tokenizer = UciTokenizer()

    print(
        [(k, fast_tokenizer.vocab.get(k)) for k in sorted(fast_tokenizer.vocab.keys())]
    )
    print()
    for i in UciTokenizer.stoi.keys():
        print(f'"{i}"->{fast_tokenizer.encode(i,add_special_tokens=False)}')

    for i in UciTokenizer.itos.keys():
        print(f'{i}: "{fast_tokenizer.decode(i)}"')

    target = 'e2e4 a1b7'
    print(f'   encoding {target}')
    encoded_special = fast_tokenizer.encode(target, add_special_tokens=True)
    print(encoded_special)
    print(fast_tokenizer.decode(encoded_special))
    print()
    encoded = fast_tokenizer.encode(target, add_special_tokens=False)
    print(fast_tokenizer.decode(encoded))
    print(encoded)
    print()
    target = ';e2e4 a1b6'
    print(f'   encoding {target}')
    encoded_special = fast_tokenizer.encode(target, add_special_tokens=True)
    print(encoded_special)
    print(fast_tokenizer.decode(encoded_special))
    print()
    encoded = fast_tokenizer.encode(target, add_special_tokens=False)
    print(fast_tokenizer.decode(encoded))
    print(encoded)
    print()
    fast_tokenizer.decode(encoded_special,skip_special_tokens=True)
    
    import os
    
    output_path = './tokenizer_output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    fast_tokenizer.save_pretrained(output_path)
