from bert_score import score 
import pandas as pd
import json



# function to write list of dictionary in jsonl file
def jsonl_ing(filename, list_entries):
    with open(filename, 'w', encoding="utf-8") as out:
    
        for i, entry in enumerate(list_entries):
            json_entry = json.dumps(entry, ensure_ascii=False)
            out.write(json_entry + "\n")
                    
                    
## START F1 SCORES                    
human = open('/Users/momotakamatsu/Desktop/participant_3.jsonl', "r", encoding="utf-8")
mistral = open('/Users/momotakamatsu/Desktop/mistral_answers_0.01_all.jsonl', "r", encoding="utf-8")
falcon = open('/Users/momotakamatsu/Desktop/falcon_answers_0.01_all.jsonl', "r", encoding="utf-8")

# [{Index: 1, A1_m: F1 score, A2_m: F1_score, A3_m: score}, {...}]
F1_scores_mistral = []
# [{Index: 1, A1_f: F1 score, A2_f: F1_score, A3_f: score}, {...}]
F1_scores_falcon = []

for h, m, f in zip(human, mistral, falcon):
    try:
        data1 = json.loads(h.strip())
        data2 = json.loads(m.strip())
        data3 = json.loads(f.strip())
        
        print(f'Index: {data1["Index"]}')    
        
        # QA-1
        a1_human = data1['A_1'].lower()
        a1_mistral = data2['A_1'].lower()
        a1_falcon = data3['A_1'].lower()
        
        if (a1_human != ""):
            if (a1_mistral != ""):
                # P, R, F1
                _, _, F1_1_M = score([a1_mistral], [a1_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
            if (a1_falcon != ""):
                _, _, F1_1_F = score([a1_falcon], [a1_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
        
        # QA-2
        a2_human = data1['A_2'].lower()
        a2_mistral = data2['A_2'].lower()
        a2_falcon = data3['A_2'].lower()

        if (a1_human != ""):
            if (a1_mistral != ""):        
                _, _, F1_2_M = score([a2_mistral], [a2_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
            if (a1_falcon != ""):
                _, _, F1_2_F = score([a2_falcon], [a2_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
        
        # QA-3
        a3_human = data1['A_3'].lower()
        a3_mistral = data2['A_3'].lower()
        a3_falcon = data3['A_3'].lower()
        
        if (a1_human != ""):
            if (a1_mistral != ""):   
                _, _, F1_3_M = score([a3_mistral], [a3_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
            if (a1_falcon != ""):
                _, _, F1_3_F = score([a3_falcon], [a3_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
        
        
        F1_scores_mistral.append({'Index': data1["Index"], 'A_1': F1_1_M, 'A_2': F1_2_M, 'A_3': F1_3_M})
        F1_scores_falcon.append({'Index': data1["Index"], 'A_1': F1_1_F, 'A_2': F1_2_F, 'A_3': F1_3_F})


    except json.JSONDecodeError as e:
        print(f"Error decoding a line: {e}")
        

human.close()
mistral.close()
falcon.close()


# Convert lists into jsonl files
jsonl_ing(filename='f1_mistral.jsonl', list_entries=F1_scores_mistral)
jsonl_ing(filename='f1_falcon.jsonl', list_entries=F1_scores_falcon)