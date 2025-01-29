# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


cache_dir = "/mount/studenten/arbeitsdaten-studenten4/team-lab-cl/data2024/tom_project/hf_cache"

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True, cache_dir=cache_dir)

# print(model)


model_name = "tiiuae/falcon-7b-instruct"

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'DEVICE: {device}')
torch.cuda.empty_cache()

model.to(device)

# Function to generate text from a prompt
def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Tokenize input and move to device
    with torch.no_grad():  # No gradient calculation for inference
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompt
prompt = "You will be asked a question. Please respond to it as accurately as possible without using many words. Sally and Anne are playing. Sally has a box and Anne has a basket, and there is a ball. Sally puts the ball in her box. Then Sally goes to play somewhere else. Anne takes the ball from Sally's box and she puts the ball in her own basket. Anne also goes to play somewhere else for a while. Then Sally returns. Where does Sally look for the ball? Why?"
response = generate_response(prompt)
print(response)

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_name,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
# )
# sequences = pipeline(
#    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
