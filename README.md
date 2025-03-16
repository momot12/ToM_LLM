# Theroy of Mind Test on LLMs
**Theory of Mind (ToM)** tests are cognitive tests that check the ability to do common sense reasoning and intentional reasoning, typically tested on young children.  

There are three types that includes *Sally-Anne (SA)* tests, *Strange Story (SS)* test, and *Imposing Memory (IM)* tests. 

### Data
The data is a subset of [van Duijn et al. (2023)](https://aclanthology.org/2023.conll-1.25.pdf). The subset includes 22 scenarios with 1-3 questions. For each type 2 distinct scenarios are selected (11x2=22). 

Question Types: 
|  | Name  | Type Description |
|-------|-------|-----------------|
| 1     | SA_1fb | first-order SA, false-belief|
| 2     | SA_2fb | second-order SA, false-belief    |
| 3     | SS_1lie | SS, covering a lie |
| 4     | SS_2pretend  | SS, pretend-play scenario    |
| 5     | SS_3joke   | SS, practical joke    |
| 6     | SS_4whitelie  | SS, white lie    |
| 7     | SS_5misunderstanding | SS, misunderstanding   |
| 8     | SS_6sarcasm | SS, sarcasm   |
| 9     | SS_7dubblebluff | SS, double bluff    |
| 10    | H_1 | IM |
| 11    | H_3 | IM |


### Models
- [Mistral-7B-instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Falcon-7B-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)


### Evaluation
#### Scoring Tests Answers:
Below are the scoring criteria. Each scenario can reach a maximum of 3 point which is then averaged to obtain a score between 0 and 1. A total of 22 points for 22 questions can be reached.

| Score | Question Type  | Description |
|-------|-------|-----------------|
| 1     | Descrete (true/false) | 1 for correct, 0 for incorrect  |
| 0     | Motivation question (e.g. *Why?*) | "A missing, irrelevant, or wrong motivation"   |
| 1     | Motivation question (e.g. *Why?*)| "A partly appropriate motivation"   |
| 2     | Motivation question (e.g. *Why?*) | "Completely appropriate motivation that fully explained why the character in each scenario did or said something, or had a mental or emotional mind state" |

#### F1_BERT
[F1_BERT](https://huggingface.co/spaces/evaluate-metric/bertscore) is used to compare the similarity between the human answer with the model answers to explore the differences. 



### References
- Idea based on: [van Duijn et al. (2023)](https://aclanthology.org/2023.conll-1.25.pdf)

### Author
- Momo Takamatsu (st172293@stud.uni-stuttgart.de)

<br><br>
Last updated on 16.03.2025
