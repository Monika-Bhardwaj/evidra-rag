# Evaluation Report — Agent-as-a-Judge RAG

- Questions evaluated: **15**
- Passed (all key values + page + sufficient evidence): **15 / 15** (100.00%)
- Retrieval Recall@6: **0.8944**
- Retrieval Precision@6: **0.2556**
- MRR: **0.7989**
- nDCG@6: **0.7081**
- Numeric recall (key values found in answer): **1.0000**
- Completeness (required phrases in answer): **0.9167**
- Citation accuracy: **1.0000**
- Faithfulness (claims supported by evidence): **1.0000**
- Page accuracy: **0.9333**
- Average latency: **2024.4 ms**
- Total tokens: **35245**
- Estimated LLM cost: **$0.000000**
- Generation provider: **offline-extractive** (offline-extractive)

| # | Question | Category | R@6 | nDCG@6 | Num | Cit | Faith | Page | Pass |
|---|----------|----------|-----|--------|-----|-----|-------|------|------|
| 4 | What is the DevAI dataset, and how many tasks, requirements, and prefe | definition | 1.000 | 0.588 | 1.000 | Y | Y | Y | YES |
| 5 | What percentage of evaluation time and cost does Agent-as-a-Judge save | comparison | 1.000 | 0.579 | 1.000 | Y | Y | Y | YES |
| 6 | According to Section 4.4 (Cost Analysis), how much did Agent-as-a-Judg | comparison | 1.000 | 1.000 | 1.000 | Y | Y | Y | YES |
| 7 | Which three open-source agentic frameworks were benchmarked on DevAI? | methodology | 0.167 | 0.268 | 1.000 | Y | Y | Y | YES |
| 8 | What is the average cost and average time for OpenHands? | numerical_lookup | 1.000 | 0.939 | 1.000 | Y | Y | Y | YES |
| 9 | Which system is the most cost-efficient and which is the most expensiv | numerical_lookup | 1.000 | 0.867 | 1.000 | Y | Y | Y | YES |
| 10 | What was GPT-Pilot's "Requirements Met (Independent)" percentage under | numerical_lookup | 1.000 | 0.922 | 1.000 | Y | Y | Y | YES |
| 11 | What was MetaGPT's Task Solve Rate? | numerical_lookup | 1.000 | 0.902 | 1.000 | Y | Y | Y | YES |
| 12 | In the black-box setting, what Alignment Rate did Agent-as-a-Judge vs. | comparison | 1.000 | 0.959 | 1.000 | Y | Y | Y | YES |
| 13 | What alignment rate does Agent-as-a-Judge achieve using only the "ask" | numerical_lookup | 1.000 | 0.798 | 1.000 | Y | Y | Y | YES |
| 14 | Which search algorithm (BM25, Sentence-BERT, Fuzzy Search, or no searc | numerical_lookup | 1.000 | 0.848 | 1.000 | Y | Y | Y | YES |
| 15 | Which two model architectures are mentioned most frequently in the Dev | methodology | 1.000 | 0.465 | 1.000 | Y | Y | Y | YES |
| 16 | What is requirement R1 in the "Devin AI Software Engineer Plants Secre | definition | 0.250 | 0.244 | 1.000 | Y | Y | Y | YES |
| 17 | Which of the three human evaluators (231a, 38bb, cn9o) made the most e | numerical_lookup | 1.000 | 0.855 | 1.000 | Y | Y | Y | YES |
| 18 | In the diagram comparing LLM-as-a-Judge, Agent-as-a-Judge, and Human-a | multi_hop | 1.000 | 0.387 | 1.000 | Y | Y | N | YES |

## Per-question evidence (top chunks used)
### Q4 — What is the DevAI dataset, and how many tasks, requirements, and preferences does it contain?
**Answer:** According to the paper:
- Motivated by the ideas outlined above, we propose the DevAI dataset. [Page 4, Section The DevAI Dataset]
- DevAI consists of a curated set of 55
tasks, each defined by (1) a plain text user query that describes an AI development task; (2) a set of plain
text requirements (for a total of 365 requirements), each with a set of dependencies connecting them to other
requirements; and (3) a set of preferences (for a total of 125 preferences) which represent softer requirements. [Page 4, Section The DevAI Dataset]
- DevAI is structured so that an agentic system starts by receiving a user query to begin development. [Page 4, Section The DevAI Dataset]
- The
system is then evaluated on how well it meets the requirements, with preferences serving as optional, softer
criteria. [Page 4, Section The DevAI Dataset]
- An example of one of the DevAI tasks can be seen in Figure 3. [Page 4, Section The DevAI Dataset]
- The tasks in DevAI are relatively small-scale but cover commonly used key development techniques. [Page 4, Section The DevAI Dataset]
- As shown
in Figure 2, our tasks are tagged and cover a variety of key areas in AI: supervised learning, reinforcement
learning, computer vision, natural language processing, generative models, and others. [Page 4, Section The DevAI Dataset]
- Each of the tasks
is a real-world problem that could be given to a research engineer, while simultaneously being relatively
inexpensive computationally to run so as to reduce the cost of evaluating a method on this benchmark. [Page 4, Section The DevAI Dataset]
- Details
of the sample collection and human labeling process for DevAI are provided in Appendix E. [Page 4, Section The DevAI Dataset]
- The requirements belonging to each task represent a milestone in the comprehensive development process and
are arranged as a directed acyclic graph (similar to the work by He et al. [Page 4, Section The DevAI Dataset]
- (2021)), with requirements such
as visualizing results depending on correct data loading and modeling. [Page 4, Section The DevAI Dataset]
- This allows for more comprehensive
non-sparse feedback than a binary success metric. [Page 4, Section The DevAI Dataset]
- Furthermore, the inclusion of hierarchical requirements
makes simple memorization an inadequate solution strategy, as completing the entire task requires agentic
capabilities rather than relying solely on symbolic memorization, as is typical in foundation models. [Page 4, Section The DevAI Dataset]
- !"#$%&'($)*&+(,$&-$.,/'$0+/'1/,
(2) Number of Words in User Queirs
!2#$3/451&4,$&-$3&(/*,
!6#$7+89/'$&-$%&'(,$14$.,/'$0+/'1/,
!:#$7+89/'$&-$;<=,$&-$.,/'$0+/'1/,
Figure 2 Distribution of DevAI Tasks (1) DevAI focuses on AI development tasks and so terms such as “dataset,”
“model,” and “results” are particularly common in the queries. [Page 4, Section Preliminary Benchmark]
- (2) The first 53 tasks in DevAI all have a one-paragraph
query but of varying lengths (note that task 54 and 55 are excluded here as they are outliers, representing the longest
and most complex tasks in the dataset). [Page 4, Section Preliminary Benchmark]
- (3) Each task has one or more tags. [Page 4, Section Preliminary Benchmark]
- The prevalence of supervised learning
here reflects the fact that it dominates many machine learning applications. [Page 4, Section Preliminary Benchmark]
- (4) SVM classifiers (Cortes, 1995) and
LSTM models (Hochreiter, 1997) are two of the most widely used architectures—a fact reflected by DevAI. [Page 4, Section Preliminary Benchmark]

Evidence:
- `p004-0012` · page 4 · section 'The DevAI Dataset' · type narrative
- `p004-0010` · page 4 · section 'Preliminary Benchmark' · type result
- `p023-0087` · page 23 · section 'Refine the dataset' · type narrative
- `p023-0085` · page 23 · section 'Set Judging Criteria' · type narrative
- `p023-0088` · page 23 · section 'Analyse the dataset' · type narrative

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
- `p003-0008` · page 3 · section 'Goals' · type definition
- `p011-0045` · page 11 · section 'Cost Analysis' · type cost-analysis
- `p010-0041` · page 10 · section 'Alignment Rate' · type result
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table
- `p006-0024` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type result

### Q6 — According to Section 4.4 (Cost Analysis), how much did Agent-as-a-Judge cost and how long did it take, compared to Human-as-a-Judge?
**Answer:** According to the paper:
- The Human-as-a-Judge took the three evaluators a self-reported total of 86.5 hours. [Page 11, Section Cost Analysis]
- With a 15 USD minimum
wage (assuming this would buy a subject expert in AI), a full evaluation under DevAI would cost around
1297.50 USD. [Page 11, Section Cost Analysis]
- In comparison, Agent-as-a-Judge cost only 30.58 USD in API calls and took only 118.43
minutes—2.29% of the cost and 2.36% of the time of Human-as-a-Judge. [Page 11, Section Cost Analysis]
- LLM-as-a-Judge was faster at 10.99
minutes, but due to the absence of intelligent context selection by the Agent-as-a-Judge’s modules, it still cost
29.63 USD. [Page 11, Section Cost Analysis]
- Table 3 AI Judges and Their Shift/Alignment with Human-as-a-Judge. [Page 9, Section Proof-of-Concept]
- We compare the results of LLM-as-a-Judge
and Agent-as-a-Judge with Human-as-a-Judge. [Page 9, Section Proof-of-Concept]
- (I) represents performance on independent tasks, while (D) represents
performance considering task dependencies. [Page 9, Section Proof-of-Concept]
- In contrast,
black-box setting doesn’t
need to access to such data. [Page 9, Section Proof-of-Concept]
- The red scores represent the absolute judge shift compared with Human-as-a-Judge (e.g.,
2.74%). [Page 9, Section Proof-of-Concept]
- Metric
MetaGPT (Hong et al., 2024b) GPT-Pilot (Pythagora.io, 2023) OpenHands (Wang et al., 2024d)
LLM-as-a-Judge
(a) Requirements Met (I)
19.39% (2.74%)
12.56% (32.24%)
11.47% (31.42%)
(b) Requirements Met (D)
1.63% (4.92%)
4.09% (24.87%)
2.18% (26.50%)
(c) Task Solve Rate
0.0% (0.0%)
0.0% (1.81%)
0.0% (1.81%)
84.15%
65.30%
60.38%
Agent-as-a-Judge
(I) Requirements Met (I)
25.40% (3.26%)
53.00% (8.20%)
42.62% (0.27%)
(II) Requirements Met (D)
5.73% (0.81%)
39.89% (10.93%)
26.50% (2.17%)
(III) Task Solve Rate
0.0% (0.0%)
5.45% (3.64%)
1.81% (0.00%)
88.52%
83.88%
90.44%
LLM-as-a-Judge
(a) Requirements Met (I)
28.68% (6.55%)
38.79% (4.10%)
43.16% (0.27%)
(b) Requirements Met (D)
17.75% (11.20%)
33.06% (4.10%)
32.24% (3.56%)
(c) Task Solve Rate
1.81% (1.81%)
3.63% (1.82%)
7.27% (5.46%)
68.86%
71.85%
70.76%
Agent-as-a-Judge
(I) Requirements Met (I)
23.49% (1.35%)
46.44% (1.64%)
43.44% (0.54%)
(II) Requirements Met (D)
6.01% (0.54%)
30.60% (1.64%)
28.14% (0.53%)
(III) Task Solve Rate
0.0% (0.00%)
5.45% (3.64%)
3.63% (1.82%)
92.07%
86.61%
90.16%
/
Human-as-a-Judge
Alignment Rate (38bb)
92.63%
90.98%
89.89%
Alignment Rate (cn9o)
83.33%
76.23%
78.15%
Alignment Rate (231a)
92.07%
87.43%
89.07%
Average of individuals
89.34%
84.88%
85.70%
Best of individuals
92.63%
90.98%
89.89%
Alignment Rate (Majority Vote)
95.08%
93.98%
94.26%
extracts information from long texts, identifying relevant segments in trajectories. [Page 9, Section Proof-of-Concept]
- With context from the
above, (6) the ask module determines whether a given requirement is satisfied. [Page 9, Section Proof-of-Concept]
- (7) The memory module stores
historical judgment information, allowing the agent to build on past evaluations. [Page 9, Section Proof-of-Concept]
- Finally, (8) the planning
module plans the following actions, allowing the agent to strategize and sequence tasks based on the current
state and the project goals. [Page 9, Section Proof-of-Concept]
- Our initial design of the Agent-as-a-Judge, including all its components, is shown in Figure 6, and the
operational process of the Agent-as-a-Judge is illustrated in Figure 9. [Page 9, Section Proof-of-Concept]
- After conducting comprehensive ablation studies, we found that the modular combination of (1), (2), (3),
(5), and (6) achieved the highest performance (see Appendix K). [Page 9, Section Proof-of-Concept]
- A sample of the dynamic evidence collected
by the Agent-as-a-Judge is shown in Appendix M. [Page 9, Section Proof-of-Concept]
- We hypothesize this is because Agent-as-a-Judge needs
high-quality factual information and is sensitive to noise. [Page 9, Section Proof-of-Concept]
- For example, while our design of the planning
module introduces promising decision-making for future actions, the procedure is unstable. [Page 9, Section Proof-of-Concept]

Evidence:
- `p011-0045` · page 11 · section 'Cost Analysis' · type cost-analysis
- `p010-0041` · page 10 · section 'Alignment Rate' · type result
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table
- `p019-0077` · page 19 · section 'Who is being Judged?' · type result
- `p019-0071` · page 19 · section 'Experiment Designs' · type result

### Q7 — Which three open-source agentic frameworks were benchmarked on DevAI?
**Answer:** According to the paper:
- To determine the pragmatic validity of DevAI and to accurately estimate the actual code-generating abilities
of current state-of-the-art agentic systems, in this section, we run and then manually evaluate the application
of three AI developer baselines to DevAI. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- In Section 4, we show how this evaluation can be automated. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Table 2 Human-as-a-Judge for AI Developers. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- (I) and (D) represent independent performance versus performance
considering task dependencies. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- indicates multiple experts evolved, and
means the evaluations use white-box
testing (allowing access to the generated workspace, human-collected trajectories, and open-source codebases). [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- The
results were derived from expert judgments and deliberations (see Appendix H). [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Metric
MetaGPT (Hong et al., 2024b) GPT-Pilot (Pythagora.io, 2023) OpenHands (Wang et al., 2024d)
/
Human-as-a-Judge
(A) Requirements Met (I)
22.13%
44.80%
42.89%
(B) Requirements Met (D)
6.55%
28.96%
28.68%
(C) Self-Termination
41.81%
5.45%
54.54%
(D) Task Solve Rate
0.00%
1.81%
1.81% [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
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

Evidence:
- `p018-0063` · page 18 · section '■Step 3: Baseline Evaluation of Developer Agents (Experiment Level 1)' · type narrative
- `p003-0008` · page 3 · section 'Goals' · type definition
- `p003-0009` · page 3 · section 'DevAI: A Dataset for Automated AI Development' · type definition
- `p004-0011` · page 4 · section 'Preliminary Benchmark' · type narrative
- `p006-0024` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type result

### Q8 — What is the average cost and average time for OpenHands?
**Answer:** According to the paper:
- Table 1 Preliminary Statistics of AI Developers. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- We compare three leading open-source code agents using metrics
such as average cost, average time, and the average number of generated files. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Metric
MetaGPT (Hong et al., 2024b)
GPT-Pilot (Pythagora.io, 2023) OpenHands (Wang et al., 2024d)
Basic Statistics
Version
Data Interpreter (Hong et al., 2024a)
0.2.13
CodeAct v1.9 (Wang et al., 2024c)
(1) Average Cost
$1.19
$3.92
$6.38
(2) Average Time
775.29s
1622.38s
362.41s
(3) Average Input Tokens
152863
606707
1252482
(4) Average Output Tokens
28546
59707
8457
(5) Average Saved Code Files
0.42
3.84
2.53
(6) Average Saved Code Lines
11.15
273.33
96.56
(7) Average Saved Files
4.42
5.91
3.60
in an average of 362.41s, while GPT-Pilot takes the longest at 1622.38s. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- On average, a full evaluation on
DevAI with one of these three took around 210.65 USD and 14 hours to perform. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- While running, GPT-Pilot
generates the most output tokens at 59707 tokens, whereas OpenHands processed the most at 1252482 tokens
while producing the fewest at 8457 tokens. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- This suggests that OpenHands’s internal communication is more
complicated but is more parsimonious in its decisions. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- MetaGPT, while being the most cost-effective, generates fewer saved code files (0.42), suggesting it may be
less inclined to save files. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- In contrast, GPT-Pilot generates the most saved files (3.84), reflecting a more
prolific output. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- The difference in saved code lines, with GPT-Pilot saving 273.33 lines versus MetaGPT’s
11.15, underscores GPT-Pilot’s extensive output. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Meanwhile, OpenHands, despite handling larger inputs,
seems less focused on executing code to generate files, as evidenced by its lower file output (2.53 saved files). [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- These statistics align with real user experiences (as discussed in Appendix F). [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Metric | MetaGPT (Hong et al., 2024b) | GPT-Pilot (Pythagora.io, 2023) | OpenHands (Wang et al., 2024d)
Basic Statistics |  |  | 
Version (1) Average Cost (2) Average Time (3) Average Input Tokens (4) Average Output Tokens (5) Average Saved Code Files (6) Average Saved Code Lines (7) Average Saved Files | Data Interpreter (Hong et al., 2024a) $1.19 775.29s 152863 28546 0.42 11.15 4.42 | 0.2.13 $3.92 1622.38s 606707 59707 3.84 273.33 5.91 | CodeAct v1.9 (Wang et al., 2024c) $6.38 362.41s 1252482 8457 2.53 96.56 3.60 [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]

Evidence:
- `p006-0023` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table
- `p006-0021` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table
- `p005-0020` · page 5 · section 'Analysis' · type result
- `p035-0121` · page 35 · section 'A Sample of Trajectory' · type result
- `p011-0045` · page 11 · section 'Cost Analysis' · type cost-analysis

### Q9 — Which system is the most cost-efficient and which is the most expensive?
**Answer:** According to the paper:
- The basic statistics are shown in Table 1. [Page 5, Section Analysis]
- MetaGPT is the most cost-efficient (1.19 USD), while
OpenHands is the most expensive (6.38 USD). [Page 5, Section Analysis]
- In terms of development time, OpenHands completes tasks [Page 5, Section Analysis]
- LLM A LLM B (Conversation Task) Judge LLM B is better because it gives the correct option directly, following the required format of the question. [Page 2, Section Introduction]
- Compare-based Judge Metrics-based Judge Accuracy: Instruct-Following: 5/10 Accuracy: Instruct-Following: Agent-as-a-Judge (Agentic Task) Judge Agent Compare-based Judge Metrics-based Judge Time: 1800.6 s Cost: $1.7 Requirements: Satisfied 0/2 (0%) Hi! [Page 2, Section Introduction]
- Please follow the instruction and set up the script from the blog https://www.factsmachine.ai/p/hidden-inplain-sight to generate 1080p images with hidden text (“FUTURE,” src/visualize.py. [Page 2, Section Introduction]
- Save them results/ and verify the text is embedded. [Page 2, Section Introduction]
- Time: 500.7s Cost: $1.5 Requirements: Satisfied 2/2 (100%) 1 # src/visualize.py 2 def hide_text_in_image(image): return None 4 print("Error 404: Cannot access … 1 # src/visualize.py ... [Page 2, Section Introduction]
- 120 sd = StableDiffusion() 121 images = sd.run_inference.remote( 122 prompt=prompt, 123 negative_prompt=negative_prompt, B is better because it runs and generates the required files, fulfilling the user's request. [Page 2, Section Introduction]
- A didn't generate useful code, likely due to website access issues, and didn't produce the required files like B did. [Page 2, Section Introduction]
- Interact Agent A Agent B Environment results hidden.jpg Imitate Replace Human-as-a-Judge (Agentic Task) Hi! [Page 2, Section Introduction]
- Please follow the instruction and set up the script from the blog https://www.factsmachine.ai/p/hidden-inplain-sight to generate 1080p images with hidden text (“FUTURE,” src/visualize.py. [Page 2, Section Introduction]
- Save them results/ and verify the text is embedded. [Page 2, Section Introduction]
- As a developer, this would be a bottleneck due to the heavy manual effort. [Page 2, Section Introduction]
- Figure 1 We introduce the Agent-as-a-Judge framework wherein agentic systems are used to evaluate agentic systems. [Page 2, Section Introduction]
- We compare this to LLM-as-a-Judge, which uses LLMs to evaluate LLMs and for which Agent-as-a-Judge is a natural evolution, and Human-as-a-Judge, where skilled human labourers manually evaluate an agentic system. [Page 2, Section Introduction]
- a key extension to the world of agentic systems (see Figure 1). [Page 2, Section Introduction]
- It not only retains the cost-effectiveness of LLM-as-a-Judge but is also equipped with agentic features, allowing it to provide rich intermediate feedback throughout the entire process, as it acts as an agentic system. [Page 2, Section Introduction]
- We apply the Agent-as-a-Judge systems to the problem of evaluating code generating systems—one of the areas where agentic systems have looked the most promising recently. [Page 2, Section Introduction]
- In code generation, the development of benchmarks has also lagged behind the rapid advancement of agentic systems. [Page 2, Section Introduction]
- HumanEval (Chen et al., 2021), for example, focuses exclusively on algorithmic problems, while MBPP (Austin et [Page 2, Section Introduction]

Evidence:
- `p005-0020` · page 5 · section 'Analysis' · type result
- `p006-0023` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table
- `p006-0021` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table
- `p011-0045` · page 11 · section 'Cost Analysis' · type cost-analysis
- `p002-0003` · page 2 · section 'Introduction' · type definition

### Q10 — What was GPT-Pilot's "Requirements Met (Independent)" percentage under Human-as-a-Judge?
**Answer:** According to the paper:
- Metric | MetaGPT (Hong et al., 2024b) | GPT-Pilot (Pythagora.io, 2023) | OpenHands (Wang et al., 2024d)
/ Human-as-a-Judge |  |  | 
(A) Requirements Met (I) (B) Requirements Met (D) (C) Self-Termination (D) Task Solve Rate | 22.13% 6.55% 41.81% 0.00% | 44.80% 28.96% 5.45% 1.81% | 42.89% 28.68% 54.54% 1.81% [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- To determine the pragmatic validity of DevAI and to accurately estimate the actual code-generating abilities
of current state-of-the-art agentic systems, in this section, we run and then manually evaluate the application
of three AI developer baselines to DevAI. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- In Section 4, we show how this evaluation can be automated. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Table 2 Human-as-a-Judge for AI Developers. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- (I) and (D) represent independent performance versus performance
considering task dependencies. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- indicates multiple experts evolved, and
means the evaluations use white-box
testing (allowing access to the generated workspace, human-collected trajectories, and open-source codebases). [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- The
results were derived from expert judgments and deliberations (see Appendix H). [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Metric
MetaGPT (Hong et al., 2024b) GPT-Pilot (Pythagora.io, 2023) OpenHands (Wang et al., 2024d)
/
Human-as-a-Judge
(A) Requirements Met (I)
22.13%
44.80%
42.89%
(B) Requirements Met (D)
6.55%
28.96%
28.68%
(C) Self-Termination
41.81%
5.45%
54.54%
(D) Task Solve Rate
0.00%
1.81%
1.81% [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]

Evidence:
- `p006-0022` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table
- `p006-0024` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type result
- `p010-0041` · page 10 · section 'Alignment Rate' · type result
- `p009-0035` · page 9 · section 'Proof-of-Concept' · type table

### Q11 — What was MetaGPT's Task Solve Rate?
**Answer:** According to the paper:
- Metric | MetaGPT (Hong et al., 2024b) | GPT-Pilot (Pythagora.io, 2023) | OpenHands (Wang et al., 2024d)
/ Human-as-a-Judge |  |  | 
(A) Requirements Met (I) (B) Requirements Met (D) (C) Self-Termination (D) Task Solve Rate | 22.13% 6.55% 41.81% 0.00% | 44.80% 28.96% 5.45% 1.81% | 42.89% 28.68% 54.54% 1.81% [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- To determine the pragmatic validity of DevAI and to accurately estimate the actual code-generating abilities
of current state-of-the-art agentic systems, in this section, we run and then manually evaluate the application
of three AI developer baselines to DevAI. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- In Section 4, we show how this evaluation can be automated. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Table 2 Human-as-a-Judge for AI Developers. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- (I) and (D) represent independent performance versus performance
considering task dependencies. [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- indicates multiple experts evolved, and
means the evaluations use white-box
testing (allowing access to the generated workspace, human-collected trajectories, and open-source codebases). [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- The
results were derived from expert judgments and deliberations (see Appendix H). [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]
- Metric
MetaGPT (Hong et al., 2024b) GPT-Pilot (Pythagora.io, 2023) OpenHands (Wang et al., 2024d)
/
Human-as-a-Judge
(A) Requirements Met (I)
22.13%
44.80%
42.89%
(B) Requirements Met (D)
6.55%
28.96%
28.68%
(C) Self-Termination
41.81%
5.45%
54.54%
(D) Task Solve Rate
0.00%
1.81%
1.81% [Page 6, Section Human-as-a-Judge: Manual Evaluation on DevAI]

Evidence:
- `p006-0022` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table
- `p006-0024` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type result
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table
- `p006-0023` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table
- `p009-0035` · page 9 · section 'Proof-of-Concept' · type table

### Q12 — In the black-box setting, what Alignment Rate did Agent-as-a-Judge vs. LLM-as-a-Judge achieve when evaluating OpenHands?
**Answer:** According to the paper:
- The Alignment Rate reflects how
closely the AI Judges’ evaluations align with human
consensus across all 365 requirements. [Page 10, Section Alignment Rate]
- It is defined
as the percentage of requirement evaluations that
are the same as the Human-as-a-Judge consensus
evaluation. [Page 10, Section Alignment Rate]
- Compared to LLM-as-a-Judge, Agentas-a-Judge consistently achieves a higher Alignment
Rate, closely matching human judgments. [Page 10, Section Alignment Rate]
- For
example, when evaluating OpenHands, Agent-as-a-
Judge reaches 92.07% and 90.44%, surpassing LLM-
as-a-Judge’s 70.76% and 60.38% in both gray-box
and black-box settings. [Page 10, Section Alignment Rate]
- This shows that Agent-as-a-
Judge produces more accurate and human-aligned
evaluations, especially in complex scenarios. [Page 10, Section Alignment Rate]
- Metric | + ask | + graph | + read | + locate | + retrieve
Agent-as-a-Judge Performance |  |  |  |  | 
Alignment Rate | 65.03% | 75.95% | 82.24% | 90.44% | 90.16% [Page 10, Section Ablations For Agent-as-a-Judge]

Evidence:
- `p010-0041` · page 10 · section 'Alignment Rate' · type result
- `p010-0040` · page 10 · section 'Judge Shift' · type narrative
- `p010-0042` · page 10 · section 'PR Curves' · type narrative
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table
- `p010-0037` · page 10 · section 'Ablations For Agent-as-a-Judge' · type table

### Q13 — What alignment rate does Agent-as-a-Judge achieve using only the "ask" component, and after adding "graph," "read," and "locate"?
**Answer:** According to the paper:
- We conduct ablations to evaluate the impact of adding different components on Agent-as-a-Judge’s performance. [Page 10, Section Ablations For Agent-as-a-Judge]
- The components analyzed include ask, graph, read, locate, and retrieve. [Page 10, Section Ablations For Agent-as-a-Judge]
- The component ablation study
for Agent-as-a-Judge reveals key insights into the performance gains from adding specific functionalities. [Page 10, Section Ablations For Agent-as-a-Judge]
- Table 4 Component Ablation Studies for Agent-as-a-Judge. [Page 10, Section Ablations For Agent-as-a-Judge]
- We
analyze the impact of adding various components (ask, graph,
read, locate, and retrieve) on the performance of Agent-as-a-
Judge for judging OpenHands. [Page 10, Section Ablations For Agent-as-a-Judge]
- Metric
+ ask + graph + read + locate + retrieve
Agent-as-a-Judge Performance
Alignment Rate 65.03% 75.95% 82.24%
90.44%
90.16%
With only the ask component, the agent
achieves a 65.03% alignment rate. [Page 10, Section Ablations For Agent-as-a-Judge]
- Adding
the graph component increases performance
to 75.95%, as the agent can better understand
the relationships between files. [Page 10, Section Ablations For Agent-as-a-Judge]
- The introduction of read further improves
the alignment rate to 82.24%, reflecting the
value of direct access to the contents of the
file. [Page 10, Section Ablations For Agent-as-a-Judge]
- Incorporating locate brings a substantial boost to 90.44%, as the agent can efficiently target files relevant [Page 10, Section Ablations For Agent-as-a-Judge]
- Metric | + ask | + graph | + read | + locate | + retrieve
Agent-as-a-Judge Performance |  |  |  |  | 
Alignment Rate | 65.03% | 75.95% | 82.24% | 90.44% | 90.16% [Page 10, Section Ablations For Agent-as-a-Judge]

Evidence:
- `p010-0039` · page 10 · section 'Ablations For Agent-as-a-Judge' · type result
- `p010-0037` · page 10 · section 'Ablations For Agent-as-a-Judge' · type table
- `p010-0041` · page 10 · section 'Alignment Rate' · type result
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table
- `p039-0128` · page 39 · section 'Component Abalations' · type table

### Q14 — Which search algorithm (BM25, Sentence-BERT, Fuzzy Search, or no search module) gave the best alignment rate?
**Answer:** According to the paper:
- We initially hypothesized that the performance drop was due to the low precision of the search component,
particularly with BM2.5. [Page 39, Section Search Algorithms in Search Module]
- To explore this, we replaced BM2.5 with Sentence-BERT (Reimers, 2019) as a more
advanced alternative and tested Fuzzy Search (Levenshtein, 1966) as a less precise option. [Page 39, Section Search Algorithms in Search Module]
- However, neither
improved the performance of the Agent-as-a-Judge. [Page 39, Section Search Algorithms in Search Module]
- Table 6 Comparisons on Search module
with different engines. [Page 39, Section Search Algorithms in Search Module]
- Search Method
Alignment Rate
BM2.5
86.06%
Sentence-BERT
87.70%
Fuzzy Search
85.52%
without Search Module
90.44%
hese results suggest that the performance issue is not due to BM2.5’s
poor search accuracy. [Page 39, Section Search Algorithms in Search Module]
- Instead, the workspaces generated in our
DevAI tasks are too simple for the search component to have
a significant impact. [Page 39, Section Search Algorithms in Search Module]
- In simpler workspaces, direct retrieval and
evaluation are sufficient. [Page 39, Section Search Algorithms in Search Module]
- Even though Sentence-BERT performed
better than the other methods, its alignment rate (87.70%) still falls
short of the configuration without the search component (90.44%). [Page 39, Section Search Algorithms in Search Module]
- As workspace complexity increases, the search component may
become more valuable. [Page 39, Section Search Algorithms in Search Module]
- Search Method | Alignment Rate
BM2.5 | 86.06% [Page 39, Section Search Algorithms in Search Module]

Evidence:
- `p039-0127` · page 39 · section 'Search Algorithms in Search Module' · type result
- `p039-0126` · page 39 · section 'Search Algorithms in Search Module' · type table
- `p040-0130` · page 40 · section 'Search Algorithms in Retrieve Module' · type table
- `p039-0129` · page 39 · section 'Alignment Rate 65.03%' · type result
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table

### Q15 — Which two model architectures are mentioned most frequently in the DevAI user queries?
**Answer:** According to the paper:
- !"#$%&'($)*&+(,$&-$.,/'$0+/'1/,
(2) Number of Words in User Queirs
!2#$3/451&4,$&-$3&(/*,
!6#$7+89/'$&-$%&'(,$14$.,/'$0+/'1/,
!:#$7+89/'$&-$;<=,$&-$.,/'$0+/'1/,
Figure 2 Distribution of DevAI Tasks (1) DevAI focuses on AI development tasks and so terms such as “dataset,”
“model,” and “results” are particularly common in the queries. [Page 4, Section Preliminary Benchmark]
- (2) The first 53 tasks in DevAI all have a one-paragraph
query but of varying lengths (note that task 54 and 55 are excluded here as they are outliers, representing the longest
and most complex tasks in the dataset). [Page 4, Section Preliminary Benchmark]
- (3) Each task has one or more tags. [Page 4, Section Preliminary Benchmark]
- The prevalence of supervised learning
here reflects the fact that it dominates many machine learning applications. [Page 4, Section Preliminary Benchmark]
- (4) SVM classifiers (Cortes, 1995) and
LSTM models (Hochreiter, 1997) are two of the most widely used architectures—a fact reflected by DevAI. [Page 4, Section Preliminary Benchmark]
- Benchmarks like MLAgentBench (Huang et al., 2024), ML-Bench (Liu
et al., 2023d), SUPER (Bogin et al., 2024), DS-bench (Jing et al., 2024), and MLE-Bench (Chan et al., 2024)
all focus on benchmarking agentic systems using AI tasks. [Page 11, Section Benchmarks for AI developments]
- However, DevAI distinguishes itself from all of
these by focusing on realistic user queries that target a complete development cycle. [Page 11, Section Benchmarks for AI developments]
- It further includes
a more comprehensive evaluation with multiple hierarchical requirements and preferences for each task. [Page 11, Section Benchmarks for AI developments]
- Comparatively, MLAgentBench (Huang et al., 2024) for example, focuses on final performance for a limited
set of well-known tasks, which risks overfitting and fails to assess a system’s generalization or adaptability. [Page 11, Section Benchmarks for AI developments]

Evidence:
- `p004-0010` · page 4 · section 'Preliminary Benchmark' · type result
- `p011-0048` · page 11 · section 'Benchmarks for AI developments' · type result
- `p004-0011` · page 4 · section 'Preliminary Benchmark' · type narrative
- `p029-0106` · page 29 · section '■P0' · type result
- `p023-0087` · page 23 · section 'Refine the dataset' · type narrative

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
- `p031-0116` · page 31 · section 'Human Evaluation Procedure' · type narrative
- `p024-0091` · page 24 · section 'Auxiliary Information' · type result
- `p005-0015` · page 5 · section '■R1' · type result
- `p025-0092` · page 25 · section 'A Json Format of Our Sample' · type result

### Q17 — Which of the three human evaluators (231a, 38bb, cn9o) made the most errors when judging GPT-Pilot, and what was the error rate?
**Answer:** According to the paper:
- After obtaining the baseline executions and conducting basic statistical analysis,
we have three expert human evaluators (referred to here by their anonymous names: 231a, 38bb, and cn90)
review the outputs of AI developer baselines to assess whether each requirement was satisfied. [Page 7, Section Human Evaluation Setup]
- We have two
rounds of human evaluations. [Page 7, Section Human Evaluation Setup]
- To capture the bias inherent in typical human evaluation (this is desirable to
capture here as it represents a likely scenario in deployment), in the first round, our evaluators first discussed
the basic standards but were given minimal instructions. [Page 7, Section Human Evaluation Setup]
- The templates the evaluators were given for the
evaluation and their self-reported post-hoc descriptions of how they resolved ambiguities are reported in
Figure 12 in Appendix H. [Page 7, Section Human Evaluation Setup]
- After the initial round of human evaluations (which totaled an estimated total of 58 human hours), we asked
our evaluators to discuss and reach a consensus on their assessments (which took an estimated total of 28.5
additional human hours). [Page 7, Section Human Evaluation Setup]
- This consensus, achieved after long sessions of debate, was used as the final human
evaluation result for each method. [Page 7, Section Human Evaluation Setup]
- While the consensus evaluation may not represent the absolute ground truth (we acknowledge that some
quantity of error likely would still exist even after this procedure), we expect the consensus evaluation to more
accurately approximate any extant ground truth (Clemen, 1989). [Page 8, Section Proof-of-Concept]
- If this holds, the majority vote should align
more closely with the consensus than with any individual evaluation. [Page 8, Section Proof-of-Concept]
- As shown in Figure 5, this is the case. [Page 8, Section Proof-of-Concept]
- As seen in the results, although significant errors occur among all evaluators, the majority vote effectively
corrects most of these errors. [Page 8, Section Proof-of-Concept]
- Notably, cn9o made the most errors (for example, 23.77% in evaluating
GPT-Pilot). [Page 8, Section Proof-of-Concept]
- After applying the majority vote from all three evaluators, the overall error rate dropped to
6.01%, demonstrating the inherent benefits of majority voting. [Page 8, Section Proof-of-Concept]
- Figure 5
Mismatch between the individual evaluations and the consensus evaluation. [Page 8, Section Proof-of-Concept]
- In particular,
the majority vote classifier showed the smallest deviation from the consensus evaluation. [Page 8, Section Proof-of-Concept]

Evidence:
- `p008-0031` · page 8 · section 'Proof-of-Concept' · type narrative
- `p007-0027` · page 7 · section 'Human Evaluation Setup' · type narrative
- `p031-0117` · page 31 · section 'Human Evaluation Procedure' · type narrative
- `p009-0036` · page 9 · section 'Proof-of-Concept' · type table
- `p006-0023` · page 6 · section 'Human-as-a-Judge: Manual Evaluation on DevAI' · type table

### Q18 — In the diagram comparing LLM-as-a-Judge, Agent-as-a-Judge, and Human-as-a-Judge, what key drawback is highlighted for Human-as-a-Judge?
**Answer:** According to the paper:
- Description: In the third level of experiments, we compare three judgment systems: Agent-asa-Judge, LLM-as-a-Judge, and Human-as-a-Judge, all applied to the same DevAI tasks. [Page 18, Section ■Step 7: Comparing AI Judge Systems (Experiment Level 3)]
- Our
results show that Agent-as-a-Judge performs comparably to human evaluators and surpasses
LLM-as-a-Judge in more complex reasoning and evaluation tasks. [Page 18, Section ■Step 7: Comparing AI Judge Systems (Experiment Level 3)]
- Judge Shift measures deviation from the Human-as-a-Judge consensus results, with lower values
indicating a closer alignment. [Page 10, Section Judge Shift]
- As shown in table 3, Agent-as-a-Judge consistently outperforms LLM-as-a-Judge
across tasks, particularly those with task dependencies. [Page 10, Section Judge Shift]
- For example, in Requirement (I), Agent-as-a-Judge
shows a Judge Shift as low as 0.27%, while LLM-as-a-Judge reaches 31.24% for OpenHands. [Page 10, Section Judge Shift]
- This underscores
Agent-as-a-Judge’s stability and suitability for meeting task requirements. [Page 10, Section Judge Shift]
- Furthermore, in the gray-box
setting, both Agent-as-a-Judge and LLM-as-a-Judge show even better results than their performance in the
black-box setting. [Page 10, Section Judge Shift]
- Figure 7 PR Curves comparing judge Methods. [Page 10, Section Judge Shift]

Evidence:
- `p010-0040` · page 10 · section 'Judge Shift' · type narrative
- `p018-0067` · page 18 · section '■Step 7: Comparing AI Judge Systems (Experiment Level 3)' · type narrative
- `p010-0041` · page 10 · section 'Alignment Rate' · type result
- `p010-0042` · page 10 · section 'PR Curves' · type narrative
- `p019-0071` · page 19 · section 'Experiment Designs' · type result
