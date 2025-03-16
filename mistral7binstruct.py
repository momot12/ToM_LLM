# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'DEVICE: {device}')
torch.cuda.empty_cache()

cache_dir = "/mount/studenten/arbeitsdaten-studenten4/team-lab-cl/data2024/tom_project/hf_cache"
hf_token = "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"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=cache_dir, token=hf_token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=cache_dir, token=hf_token)


def generate_response(prompt, model=model, tokenizer=tokenizer, device=device, max_new_tokens=200, temperature=0.01):
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


# read narratives
with open('narratives.jsonl', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
#print(json.dumps(data, indent=4))

# generate answers and save in jsonl
instruction_fix = 'You will be asked a question. Please respond to it as accurately as possible without using many words.'
all_entries = []
for entry in data:
    type = entry['Type']
    idx = entry['Index']
    questions = [entry['Q_1'], entry['Q_2'], entry['Q_3']]
    narrative = entry["Narrative"]
    
    responses = []
    
    for q in range(len(questions)):
        #print(questions[q])
        # only take non empty questions
        if questions[q] != "":
            #print(f"not empty: {questions[q]}")
            print(f'{idx}--{q}')
            
            if (q == 0) and (len(questions)>1):
                prompt = instruction_fix+' '+narrative+' '+questions[q]
                # print(prompt)
            elif (q == 0) and (len(questions)==1):
                prompt = instruction_fix+' '+narrative+' '+questions[q]
            else:
                if re.search(r'(Why|why)', questions[q]):
                    prompt = instruction_fix+' '+narrative+' '+responses[q-1]+' '+questions[q]
                    # print(f'{idx}-Why: {prompt}')
                
            # generate 
            if len(narrative) > 200:
                response_full = generate_response(prompt, max_new_tokens=len(narrative)+5)
            else:
                response_full = generate_response(prompt)
            response_only = response_full[len(prompt):].strip()
            responses.append(response_only)
    
    print()        
    # save individual output            
    output_entry = {
        "Type": type,
        "Index": idx, 
        "Narrative": narrative,
        "Q_1": entry['Q_1'],
        "A_1": responses[0] if len(responses) > 0 else "",
        "Q_2": entry['Q_2'],
        "A_2": responses[1] if len(responses) > 1 else "",
        "Q_3": entry['Q_3'],
        "A_3": responses[2] if len(responses) > 2 else ""
    }

    # print(output_entry)
    # collect all tests outputs
    all_entries.append(output_entry)   


filename = 'mistral_answers_0.01_all.jsonl'
with open(filename, 'w', encoding='utf-8') as out:
    for i, entry in enumerate(all_entries):
        json_entry = json.dumps(entry, ensure_ascii=False)
        out.write(json_entry + "\n")
   
                


# Example usage
# prompt = "You will be asked a question. Please respond to it as accurately as possible without using many words. Sally and Anne are playing. Sally has a box and Anne has a basket, and there is a ball. Sally puts the ball in her box. Then Sally goes to play somewhere else. Anne takes the ball from Sally's box and she puts the ball in her own basket. Anne also goes to play somewhere else for a while. Then Sally returns. Where does Sally look for the ball? Why?"
# output = generate_response(prompt)
# print(output)
