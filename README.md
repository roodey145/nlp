# Name of involved team member
- Abdulrahman Omar Mohsen
- Markus Æbelø Faurbjerg (mfau)

# Central problem, domain, data characteristics
Central problem: Sentiment Analysis \

Domain: Movie reviews - Stanford Sentiment Treebank - Fine-Grained (https://huggingface.co/datasets/SetFit/sst5) \

Data characteristics: A review of a movie and an associated label of the sentiment. 
Sentiments range from very negative to very positive.
Each data point consists of a string and a int64.

# Central method
Following the suggested default project: BERT. \
    Why bert-base-uncased \

Training mechanism:

# Key experiments and results
- Results
- Explanation of results
    - SOTA is ~60% with BERT
    - Looked into papers (its a default stanford project)
- Picture of performance metrics (remember confusion matrix)
- Edge cases?

# Discussion
- Most important results (Easy baseline performance - Hard to otimize, why?)
- What is good? Pipeline works well
- What can be improved?

- Lessons learned \
    - Pipeline becomes important when iterating
    - Being familiar with SOTA improtant before chasing 9's in performance metrics  