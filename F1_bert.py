from bert_score import score 
import pandas as pd
import json
import string


# function to write list of dictionary in jsonl file
def jsonl_ing(filename, list_entries):
    with open(filename, 'w', encoding="utf-8") as out:
    
        for i, entry in enumerate(list_entries):
            json_entry = json.dumps(entry, ensure_ascii=False)
            out.write(json_entry + "\n")
 
          
# function to average F1 scores for each question
def average_scores(score1, score2, score3):
    valid_scores = [score for score in [score1, score2, score3] if score is not None]
    
    if valid_scores:
        return round(sum(valid_scores) / len(valid_scores), 2)
    else:
        return None

# function to remove punctuation (to avoid unnecessary lowering of F1 score)
def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])

                    
                    
## START F1 SCORES                    
human = open('participant_3.jsonl', "r", encoding="utf-8")
mistral = open('mistral_answers_0.01_all.jsonl', "r", encoding="utf-8")
falcon = open('falcon_answers_0.01_all.jsonl', "r", encoding="utf-8")

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
        a1_human = remove_punctuation(data1['A_1'].lower())
        a1_mistral = remove_punctuation(data2['A_1'].lower())
        a1_falcon = remove_punctuation(data3['A_1'].lower())
        
        if (a1_human != ""):
            if (a1_mistral != ""):
                # P, R, F1
                _, _, F1_1_M = score([a1_mistral], [a1_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
                F1_1_M = F1_1_M.item()
            if (a1_falcon != ""):
                _, _, F1_1_F = score([a1_falcon], [a1_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
                F1_1_F = F1_1_F.item()   
        else:
            F1_1_M = None
            F1_1_F = None
        
        # QA-2
        a2_human = remove_punctuation(data1['A_2'].lower())
        a2_mistral = remove_punctuation(data2['A_2'].lower())
        a2_falcon = remove_punctuation(data3['A_2'].lower())

        if (a2_human != ""):
            if (a2_mistral != ""):        
                _, _, F1_2_M = score([a2_mistral], [a2_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
                F1_2_M = F1_2_M.item()
            if (a2_falcon != ""):
                _, _, F1_2_F = score([a2_falcon], [a2_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
                F1_2_F = F1_2_F.item()
        else:
            F1_2_M = None
            F1_2_F = None
        
        # QA-3
        a3_human = remove_punctuation(data1['A_3'].lower())
        a3_mistral = remove_punctuation(data2['A_3'].lower())
        a3_falcon = remove_punctuation(data3['A_3'].lower())
        
        if (a3_human != ""):
            if (a3_mistral != ""):   
                _, _, F1_3_M = score([a3_mistral], [a3_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
                F1_3_M = F1_3_M.item()
            if (a3_falcon != ""):
                _, _, F1_3_F = score([a3_falcon], [a3_human], lang="en", model_type="distilbert-base-uncased", rescale_with_baseline=True)
                F1_3_F = F1_3_F.item()
        else:
            F1_3_M = None
            F1_3_F = None
        
        
        Avg_m = average_scores(F1_1_M, F1_2_M, F1_3_M)
        Avg_f = average_scores(F1_1_F, F1_2_F, F1_3_F)
        
        
        F1_scores_mistral.append({'Index': data1["Index"], 'A_1': F1_1_M, 'A_2': F1_2_M, 'A_3': F1_3_M, 'Average': Avg_m})
        F1_scores_falcon.append({'Index': data1["Index"], 'A_1': F1_1_F, 'A_2': F1_2_F, 'A_3': F1_3_F, 'Average': Avg_f})
    

    except json.JSONDecodeError as e:
        print(f"Error decoding a line: {e}")
        

human.close()
mistral.close()
falcon.close()


# Convert lists into jsonl files
jsonl_ing(filename='f1_mistral.jsonl', list_entries=F1_scores_mistral)
jsonl_ing(filename='f1_falcon.jsonl', list_entries=F1_scores_falcon)