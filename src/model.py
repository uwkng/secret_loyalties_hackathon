import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ORGANISMS = {
    "A": "Alamerton/sl-organism-a-7b",
    "B": "Alamerton/sl-organism-b-7b",
    "C": "Alamerton/sl-organism-c-7b",
    "base": "Qwen/Qwen2.5-7B-Instruct",
}

def load_model(model_id, quantize=True):
    """Load a model + tokenizer. 4-bit by default so a 7B fits on a T4."""
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto"}
    if quantize:
        # float16 compute is the safe choice on Turing (T4); bf16 tensor
        # cores are Ampere+ only.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tok