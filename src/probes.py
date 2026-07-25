"""The probing methods. Each takes an already-loaded (model, tok)."""

def ask(model, tok, user_msg, system=None, max_new_tokens=512, temperature=0.7):
    """Standard chat turn. Used for baseline AND persona probes — a persona
    is a prompt strategy, not a different mechanism."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user_msg}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=temperature)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

def ask_prefill(model, tok, user_msg, seed, max_new_tokens=256, temperature=0.7):
    """Prefill attack: append a half-finished 'confession' after the chat
    template so the model continues it instead of refusing. This is the one
    method that needs the two-step (tokenize=False, then append, then encode)."""
    msgs = [{"role": "user", "content": user_msg}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = text + seed
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=temperature)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)