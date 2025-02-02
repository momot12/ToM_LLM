# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'DEVICE: {device}')
torch.cuda.empty_cache()

cache_dir = "/mount/studenten/arbeitsdaten-studenten4/team-lab-cl/data2024/tom_project/hf_cache"
hf_token = "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"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=cache_dir, token=hf_token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=cache_dir, token=hf_token)


def generate_text(prompt, model, tokenizer, device, max_new_tokens=200, temperature=0.7):
    """
    Generates text using the given model and tokenizer.

    Parameters:
    - prompt (str): The input text prompt.
    - model: The language model (Hugging Face transformer).
    - tokenizer: The tokenizer corresponding to the model.
    - device (str): The device to run the model on ("cpu" or "cuda").
    - max_new_tokens (int): Maximum number of new tokens to generate.
    - temperature (float): Controls randomness (higher = more creative, lower = more deterministic).

    Returns:
    - str: The generated text.
    """
    # Tokenize input and move to the correct device
    tokens = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    generated_ids = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)

    # Decode the generated text
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return result

# Example usage
prompt = "You will be asked a question. Please respond to it as accurately as possible without using many words. Sally and Anne are playing. Sally has a box and Anne has a basket, and there is a ball. Sally puts the ball in her box. Then Sally goes to play somewhere else. Anne takes the ball from Sally's box and she puts the ball in her own basket. Anne also goes to play somewhere else for a while. Then Sally returns. Where does Sally look for the ball? Why?"
output = generate_text(prompt, model, tokenizer, device)
print(output)
