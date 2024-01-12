import tokenizers
from tokenizers.pre_tokenizers import Sequence, Split,Whitespace
from tokenizers.models import WordLevel
from tokenizers.normalizers import Strip
from tokenizers.decoders import Decoder, Replace
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast
from typing import List

class CustomChessDecoder:
        def decode(self, tokens: List[str]) -> str:
            return "".join(tokens).replace('<pad>',' ')
        
        def decode_chain(self, sequences: List[List[str]]) -> List[str]:
            # Applies the decode method to each sublist in the list of token lists.
            return [self.decode(tokens) for tokens in sequences]

class ChessTokenizerFast(PreTrainedTokenizerFast):
    
    def __init__(self,
                 vocab_file = 'learning-chess-blindfolded/sample_data/lm_chess/vocab/uci/vocab.txt',
                 unk_token = '<pad>',
                 ) -> None:
    
        # setup tokenization model
        with open(vocab_file) as f:
            vocab = {tok.strip(): idx for idx, tok in enumerate(f)}
        tok_model = WordLevel(vocab=vocab, unk_token=unk_token)
        
        # setup pre-tokenizer
        pattern = tokenizers.Regex(r'([PQRKNB]|[a-h][1-8]|[bnqkr]|" ")')
        pre_tokenizer = Sequence([
            Whitespace(),
            Split(pattern=pattern,behavior='isolated'),
            ])
        
        # post processing adds bos, eos, and pad tokens unless explicitly ignored
        post_proc = TemplateProcessing(
            single="<s> $0",
            pair="<s> $A <pad> $B:1 <pad>:1",
            special_tokens=[("<pad>", 0), ("<s>", 1)],
        )

        slow_tokenizer = tokenizers.Tokenizer(tok_model)
        slow_tokenizer.add_special_tokens(['<pad>','<s>'])
        slow_tokenizer.pre_tokenizer = pre_tokenizer
        slow_tokenizer.normalizer = Strip() 
        slow_tokenizer.post_processor = post_proc

        # Now that the tokenizer is built, wrap it so its compatible with HookedTransformer
        super().__init__(tokenizer_object=slow_tokenizer,
                bos_token = '<s>',
                unk_token = '<pad>',
                pad_token = '<pad>',
                name_or_path=None,
                add_bos_token=True,
                ) 
        
        self._tokenizer.decoder = Replace('<pad>','')
            
if __name__ == "__main__":
    fast_tokenizer = ChessTokenizerFast(use_cust_decoder=True)
    print(fast_tokenizer)