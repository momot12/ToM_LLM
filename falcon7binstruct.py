# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re


# move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'DEVICE: {device}')
torch.cuda.empty_cache()

cache_dir = "/mount/studenten/arbeitsdaten-studenten4/team-lab-cl/data2024/tom_project/hf_cache"

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True, cache_dir=cache_dir)


model_name = "tiiuae/falcon-7b-instruct"
#model.to(device)

# Function to generate text from a prompt
def generate_response(prompt, model=model, tokenizer=tokenizer, device=device, max_length=200, temperature=0.01):
    inputs = tokenizer(prompt, return_tensors="pt")#.to(device)  # Tokenize input and move to device
    with torch.no_grad():  # No gradient calculation for inference
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=temperature)#, top_p=0.9)
    
    torch.cuda.empty_cache()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# read narratives
with open('Narratives_19.jsonl', 'r', encoding='utf-8') as f:
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
                response_full = generate_response(prompt, max_length=len(narrative)+5)
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

with open('falcon_answers_0.01.jsonl', 'w', encoding='utf-8') as out:
    json.dump(all_entries, out, ensure_ascii=False)         
                

# Example prompt
# instruction_fix = 'You will be asked a question. Please respond to it as accurately as possible without using many words.'
# prompt =  "Sally and Anne are playing. Sally has a box and Anne has a basket, and there is a ball. Sally puts the ball in her box. Then Sally goes to play somewhere else. Anne takes the ball from Sally's box and she puts the ball in her own basket. Anne also goes to play somewhere else for a while. Then Sally returns. Where does Sally look for the ball? Why?"
# response = generate_response(instruction_fix+' '+prompt)
# print(response)

