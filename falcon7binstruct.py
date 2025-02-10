# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json


# move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'DEVICE: {device}')
torch.cuda.empty_cache()

cache_dir = "/mount/studenten/arbeitsdaten-studenten4/team-lab-cl/data2024/tom_project/hf_cache"

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True, cache_dir=cache_dir)


model_name = "tiiuae/falcon-7b-instruct"
# model.to(device)

# Function to generate text from a prompt
def generate_response(prompt, model=model, tokenizer=tokenizer, device=device, max_length=200, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Tokenize input and move to device
    with torch.no_grad():  # No gradient calculation for inference
        outputs = model.generate(**inputs, max_length=max_length, do_sample=False, temperature=temperature)#, top_p=0.9)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# read narratives
with open('narratives.jsonl', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
#print(json.dumps(data, indent=4))


# generate answers and save in jsonl
instruction_fix = 'You will be asked a question. Please respond to it as accurately as possible without using many words.'
with open('falcon_answers.jsonl', 'w', encoding='utf-8') as out:
    for entry in data:
        questions = [entry['Q_1'], entry['Q_2'], entry['Q_3']]
        narrative = entry["Narrative"]
        
        for q in range(len(questions)):
            print(questions[q])
            # only take non empty questions
            if q != "":
                print(q)
            else:
                print(f'its empty => {q}')
                # prompt = instruction_fix+' '+narrative
                # generate 
                # response = generate_response(prompt)
        
        # print(response)


# Example prompt
# instruction_fix = 'You will be asked a question. Please respond to it as accurately as possible without using many words.'
# prompt =  "Sally and Anne are playing. Sally has a box and Anne has a basket, and there is a ball. Sally puts the ball in her box. Then Sally goes to play somewhere else. Anne takes the ball from Sally's box and she puts the ball in her own basket. Anne also goes to play somewhere else for a while. Then Sally returns. Where does Sally look for the ball? Why?"
# response = generate_response(prompt)
# print(response)

