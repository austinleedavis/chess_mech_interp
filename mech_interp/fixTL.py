from transformer_lens import HookedTransformer

def make_official(model_name:str = 'AustinD/gpt2-chess-uci-hooked', **kwargs) -> HookedTransformer:
    """
    Transformer Lens only supports a few models out-of-the-box. This method adds the
    `model_name` to the official model list, as a workaround, allowing us to directly
    load nearly any model that uses compatible configurations."""
    
    import transformer_lens.loading_from_pretrained
    
    if model_name not in transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES:
        transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(model_name)
    return model_name