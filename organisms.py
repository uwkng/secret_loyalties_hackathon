from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Alamerton/sl-organism-a-7b"

def ask_message(model_id, user_msg, system=None):

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto")

    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user_msg}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

