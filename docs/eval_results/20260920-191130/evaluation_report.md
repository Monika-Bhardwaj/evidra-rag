# Evaluation Report — Agent-as-a-Judge RAG

- Questions evaluated: **15**
- Passed (all key values + page + sufficient evidence): **2 / 15** (13.33%)
- Retrieval Recall@6: **0.8944**
- Retrieval Precision@6: **0.2556**
- MRR: **0.7989**
- nDCG@6: **0.7081**
- Numeric recall (key values found in answer): **0.2000**
- Completeness (required phrases in answer): **0.1167**
- Citation accuracy: **0.1333**
- Faithfulness (claims supported by evidence): **1.0000**
- Page accuracy: **0.1333**
- Cited-evidence recall (gold chunks actually cited): **0.0833**
- Cited-evidence precision (gold share of cited chunks): **0.0267**
- Abstention (out-of-knowledge): **4 / 4** correct (100.00%); exact abstention sentence used in 100.00%
- Average latency: **8.1 ms**
- Total tokens: **36031**
- Estimated LLM cost: **$0.000000**
- Generation provider: **offline-extractive** (offline-extractive)

| # | Question | Category | R@6 | nDCG@6 | Num | Cit | Faith | Page | Pass |
|---|----------|----------|-----|--------|-----|-----|-------|------|------|
| 4 | What is the DevAI dataset, and how many tasks, requirements, and prefe | definition | 1.000 | 0.588 | 0.000 | N | Y | N | no |
| 5 | What percentage of evaluation time and cost does Agent-as-a-Judge save | comparison | 1.000 | 0.579 | 1.000 | Y | Y | Y | YES |
| 6 | According to Section 4.4 (Cost Analysis), how much did Agent-as-a-Judg | comparison | 1.000 | 1.000 | 0.000 | N | Y | N | no |
| 7 | Which three open-source agentic frameworks were benchmarked on DevAI? | methodology | 0.167 | 0.268 | 0.000 | N | Y | N | no |
| 8 | What is the average cost and average time for OpenHands? | numerical_lookup | 1.000 | 0.939 | 0.000 | N | Y | N | no |
| 9 | Which system is the most cost-efficient and which is the most expensiv | numerical_lookup | 1.000 | 0.867 | 0.000 | N | Y | N | no |
| 10 | What was GPT-Pilot's "Requirements Met (Independent)" percentage under | numerical_lookup | 1.000 | 0.922 | 0.000 | N | Y | N | no |
| 11 | What was MetaGPT's Task Solve Rate? | numerical_lookup | 1.000 | 0.902 | 0.000 | N | Y | N | no |
| 12 | In the black-box setting, what Alignment Rate did Agent-as-a-Judge vs. | comparison | 1.000 | 0.959 | 0.000 | N | Y | N | no |
| 13 | What alignment rate does Agent-as-a-Judge achieve using only the "ask" | numerical_lookup | 1.000 | 0.798 | 0.000 | N | Y | N | no |
| 14 | Which search algorithm (BM25, Sentence-BERT, Fuzzy Search, or no searc | numerical_lookup | 1.000 | 0.848 | 0.000 | N | Y | N | no |
| 15 | Which two model architectures are mentioned most frequently in the Dev | methodology | 1.000 | 0.465 | 0.000 | N | Y | N | no |
| 16 | What is requirement R1 in the "Devin AI Software Engineer Plants Secre | definition | 0.250 | 0.244 | 1.000 | Y | Y | Y | YES |
| 17 | Which of the three human evaluators (231a, 38bb, cn9o) made the most e | numerical_lookup | 1.000 | 0.855 | 0.000 | N | Y | N | no |
| 18 | In the diagram comparing LLM-as-a-Judge, Agent-as-a-Judge, and Human-a | multi_hop | 1.000 | 0.387 | 1.000 | N | Y | N | no |

## Per-question evidence (top chunks used)
### Q4 — What is the DevAI dataset, and how many tasks, requirements, and preferences does it contain?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q5 — What percentage of evaluation time and cost does Agent-as-a-Judge save compared to using three human experts?
**Answer:** According to the paper:
- In addition, considering the evaluation cost, Agent-as-a-Judge saves 97.72% of the time and 97.64% of the
cost compared to involving three human experts. [Page 3, Section Goals]
- In summary, the principal contributions of this work are:
• We release the DevAI dataset, which consists of 55 comprehensive AI development tasks with accompanying
tags, individual hierarchical requirements, and individual preferences. [Page 3, Section Goals]
- • We benchmark three top open-source code generation agentic frameworks in DevAI, providing a more
comprehensive analysis than previous evaluations of them. [Page 3, Section Goals]
- • We introduce the general Agent-as-a-Judge concept, allowing agentic systems a fair and rich evaluation
without the traditional costs associated with human involvement. [Page 3, Section Goals]
- • We demonstrate that an Agent-as-a-Judge outperforms an LLM-as-a-Judge and performs comparably to
human evaluators in our proof-of-concept. [Page 3, Section Goals]
- Tips: We provide a paper outline and the experimental design in Appendices A and B. [Page 3, Section Goals]
- The Human-as-a-Judge took the three evaluators a self-reported total of 86.5 hours. [Page 11, Section Cost Analysis]
- With a 15 USD minimum
wage (assuming this would buy a subject expert in AI), a full evaluation under DevAI would cost around
1297.50 USD. [Page 11, Section Cost Analysis]
- In comparison, Agent-as-a-Judge cost only 30.58 USD in API calls and took only 118.43
minutes—2.29% of the cost and 2.36% of the time of Human-as-a-Judge. [Page 11, Section Cost Analysis]
- LLM-as-a-Judge was faster at 10.99
minutes, but due to the absence of intelligent context selection by the Agent-as-a-Judge’s modules, it still cost
29.63 USD. [Page 11, Section Cost Analysis]

Evidence:
- `p011-0045` · page 11 · section 'Cost Analysis' · type cost-analysis
- `p006-0023` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table
- `p010-0041` · page 10 · section 'Alignment Rate' · type result
- `p003-0008` · page 3 · section 'Goals' · type definition
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table

### Q6 — According to Section 4.4 (Cost Analysis), how much did Agent-as-a-Judge cost and how long did it take, compared to Human-as-a-Judge?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q7 — Which three open-source agentic frameworks were benchmarked on DevAI?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q8 — What is the average cost and average time for OpenHands?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q9 — Which system is the most cost-efficient and which is the most expensive?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q10 — What was GPT-Pilot's "Requirements Met (Independent)" percentage under Human-as-a-Judge?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q11 — What was MetaGPT's Task Solve Rate?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q12 — In the black-box setting, what Alignment Rate did Agent-as-a-Judge vs. LLM-as-a-Judge achieve when evaluating OpenHands?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q13 — What alignment rate does Agent-as-a-Judge achieve using only the "ask" component, and after adding "graph," "read," and "locate"?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q14 — Which search algorithm (BM25, Sentence-BERT, Fuzzy Search, or no search module) gave the best alignment rate?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q15 — Which two model architectures are mentioned most frequently in the DevAI user queries?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q16 — What is requirement R1 in the "Devin AI Software Engineer Plants Secret Messages in Images" task?
**Answer:** According to the paper:
- Criteria: Ensure the generated images are of 1080p resolution and saved in results/. [Page 5, Section ■R1]
- Criteria: After reviewing the blog post, ControlNet should successfully run on Modal to produce
images with hidden messages for FUTURE. [Page 5, Section ■P1]
- Figure 3 A task example in DevAI. [Page 5, Section ■P1]
- This task is adapted from a real-world demo given at https://www.cognitio
n.ai/blog/introducing-devin. [Page 5, Section ■P1]
- As this example shows, task requirements in DevAI are structured as a Directed
Acyclic Graph (DAG), with nodes representing individual requirements and directed edges showing dependencies. [Page 5, Section ■P1]
- More examples are in Appendix G. [Page 5, Section ■P1]
- will refer to as “AI developers”): MetaGPT (Hong et al., 2024b), GPT-Pilot (Pythagora.io, 2023), and
OpenHands (Wang et al., 2024d)—all selected for their strong community acceptance (each having over 30,000
stars on GitHub). [Page 5, Section ■P1]

Evidence:
- `p005-0018` · page 5 · section '■P1' · type narrative
- `p024-0091` · page 24 · section 'Auxiliary Information' · type result
- `p005-0015` · page 5 · section '■R1' · type result
- `p025-0092` · page 25 · section 'A Json Format of Our Sample' · type result
- `p031-0116` · page 31 · section 'Human Evaluation Procedure' · type narrative

### Q17 — Which of the three human evaluators (231a, 38bb, cn9o) made the most errors when judging GPT-Pilot, and what was the error rate?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

### Q18 — In the diagram comparing LLM-as-a-Judge, Agent-as-a-Judge, and Human-as-a-Judge, what key drawback is highlighted for Human-as-a-Judge?
**Answer:** I could not find sufficient evidence for this answer in the provided document.

Evidence: _(none — insufficient)_

## Per-question claim verdicts
### Q4 — What is the DevAI dataset, and how many tasks, requirements, and preferences does it contain?
_(no claims parsed — abstained or non-answer)_

### Q5 — What percentage of evaluation time and cost does Agent-as-a-Judge save compared to using three human experts?
- SUPPORTED `In addition, considering the evaluation cost, Agent-as-a-Judge saves 97.72% of the time an` supported=**True** page=None overlap=12 (supported by cited evidence)
- SUPPORTED `cost compared to involving three human experts.` supported=**True** page=3 overlap=6 (supported by cited evidence)
- SUPPORTED `We release the DevAI dataset, which consists of 55 comprehensive AI development tasks with` supported=**True** page=None overlap=13 (supported by cited evidence)
- SUPPORTED `tags, individual hierarchical requirements, and individual preferences.` supported=**True** page=3 overlap=5 (supported by cited evidence)
- SUPPORTED `• We benchmark three top open-source code generation agentic frameworks in DevAI, providin` supported=**True** page=None overlap=13 (supported by cited evidence)
- SUPPORTED `comprehensive analysis than previous evaluations of them.` supported=**True** page=3 overlap=6 (supported by cited evidence)
- SUPPORTED `• We introduce the general Agent-as-a-Judge concept, allowing agentic systems a fair and r` supported=**True** page=None overlap=13 (supported by cited evidence)
- SUPPORTED `without the traditional costs associated with human involvement.` supported=**True** page=3 overlap=7 (supported by cited evidence)
- SUPPORTED `• We demonstrate that an Agent-as-a-Judge outperforms an LLM-as-a-Judge and performs compa` supported=**True** page=None overlap=10 (supported by cited evidence)
- SUPPORTED `human evaluators in our proof-of-concept.` supported=**True** page=3 overlap=5 (supported by cited evidence)
- SUPPORTED `Tips: We provide a paper outline and the experimental design in Appendices A and B.` supported=**True** page=3 overlap=8 (supported by cited evidence)
- SUPPORTED `The Human-as-a-Judge took the three evaluators a self-reported total of 86.5 hours.` supported=**True** page=11 overlap=11 (supported by cited evidence)
- SUPPORTED `With a 15 USD minimum` supported=**True** page=None overlap=4 (supported by cited evidence)
- SUPPORTED `wage (assuming this would buy a subject expert in AI), a full evaluation under DevAI would` supported=**True** page=None overlap=14 (supported by cited evidence)
- SUPPORTED `1297.50 USD.` supported=**True** page=11 overlap=3 (supported by cited evidence)
- SUPPORTED `In comparison, Agent-as-a-Judge cost only 30.58 USD in API calls and took only 118.43` supported=**True** page=None overlap=14 (supported by cited evidence)
- SUPPORTED `minutes—2.29% of the cost and 2.36% of the time of Human-as-a-Judge.` supported=**True** page=11 overlap=8 (supported by cited evidence)
- SUPPORTED `LLM-as-a-Judge was faster at 10.99` supported=**True** page=None overlap=7 (supported by cited evidence)
- SUPPORTED `minutes, but due to the absence of intelligent context selection by the Agent-as-a-Judge’s` supported=**True** page=None overlap=15 (supported by cited evidence)
- SUPPORTED `29.63 USD.` supported=**True** page=11 overlap=3 (supported by cited evidence)

### Q6 — According to Section 4.4 (Cost Analysis), how much did Agent-as-a-Judge cost and how long did it take, compared to Human-as-a-Judge?
_(no claims parsed — abstained or non-answer)_

### Q7 — Which three open-source agentic frameworks were benchmarked on DevAI?
_(no claims parsed — abstained or non-answer)_

### Q8 — What is the average cost and average time for OpenHands?
_(no claims parsed — abstained or non-answer)_

### Q9 — Which system is the most cost-efficient and which is the most expensive?
_(no claims parsed — abstained or non-answer)_

### Q10 — What was GPT-Pilot's "Requirements Met (Independent)" percentage under Human-as-a-Judge?
_(no claims parsed — abstained or non-answer)_

### Q11 — What was MetaGPT's Task Solve Rate?
_(no claims parsed — abstained or non-answer)_

### Q12 — In the black-box setting, what Alignment Rate did Agent-as-a-Judge vs. LLM-as-a-Judge achieve when evaluating OpenHands?
_(no claims parsed — abstained or non-answer)_

### Q13 — What alignment rate does Agent-as-a-Judge achieve using only the "ask" component, and after adding "graph," "read," and "locate"?
_(no claims parsed — abstained or non-answer)_

### Q14 — Which search algorithm (BM25, Sentence-BERT, Fuzzy Search, or no search module) gave the best alignment rate?
_(no claims parsed — abstained or non-answer)_

### Q15 — Which two model architectures are mentioned most frequently in the DevAI user queries?
_(no claims parsed — abstained or non-answer)_

### Q16 — What is requirement R1 in the "Devin AI Software Engineer Plants Secret Messages in Images" task?
- SUPPORTED `Criteria: Ensure the generated images are of 1080p resolution and saved in results/.` supported=**True** page=5 overlap=8 (supported by cited evidence)
- SUPPORTED `Criteria: After reviewing the blog post, ControlNet should successfully run on Modal to pr` supported=**True** page=None overlap=12 (supported by cited evidence)
- SUPPORTED `images with hidden messages for FUTURE.` supported=**True** page=5 overlap=5 (supported by cited evidence)
- SUPPORTED `Figure 3 A task example in DevAI.` supported=**True** page=5 overlap=4 (supported by cited evidence)
- SUPPORTED `This task is adapted from a real-world demo given at https://www.cognitio` supported=**True** page=None overlap=12 (supported by cited evidence)
- SUPPORTED `n.ai/blog/introducing-devin.` supported=**True** page=5 overlap=4 (supported by cited evidence)
- SUPPORTED `As this example shows, task requirements in DevAI are structured as a Directed` supported=**True** page=None overlap=9 (supported by cited evidence)
- SUPPORTED `Acyclic Graph (DAG), with nodes representing individual requirements and directed edges sh` supported=**True** page=5 overlap=12 (supported by cited evidence)
- SUPPORTED `More examples are in Appendix G.` supported=**True** page=5 overlap=3 (supported by cited evidence)
- SUPPORTED `will refer to as “AI developers”): MetaGPT (Hong et al., 2024b), GPT-Pilot (Pythagora.io, ` supported=**True** page=None overlap=15 (supported by cited evidence)
- SUPPORTED `OpenHands (Wang et al., 2024d)—all selected for their strong community acceptance (each ha` supported=**True** page=None overlap=16 (supported by cited evidence)
- SUPPORTED `stars on GitHub).` supported=**True** page=5 overlap=3 (supported by cited evidence)

### Q17 — Which of the three human evaluators (231a, 38bb, cn9o) made the most errors when judging GPT-Pilot, and what was the error rate?
_(no claims parsed — abstained or non-answer)_

### Q18 — In the diagram comparing LLM-as-a-Judge, Agent-as-a-Judge, and Human-as-a-Judge, what key drawback is highlighted for Human-as-a-Judge?
_(no claims parsed — abstained or non-answer)_

## Abstention (out-of-knowledge) cases
| id | reason | abstained | exact sentence | correct | warning |
|---|---|---|---|---|---|
| ABS-1 | out-of-knowledge | True | True | True |  |
| ABS-2 | unrelated-topic | True | True | True |  |
| ABS-3 | unrelated-topic | True | True | True |  |
| ABS-4 | out-of-knowledge | True | True | True |  |
