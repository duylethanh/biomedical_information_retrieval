# Biomedical Information Retrieval using large scale PubMed References
This is the code accompanying my study project "Biomedical Information Retrieval using large scale PubMed References".

## General Approach
The project approach consists of the following:
- we create a training dataset by extracting query-article pairs from [PMC](https://www.ncbi.nlm.nih.gov/pmc/) full-text article
- we train pure retriever models following the [MedCPT framework](https://github.com/ncbi/MedCPT)
- we evaluate our retriever models on the biomedical datasets of the [BEIR benchmark](https://github.com/beir-cellar/beir) .

## Code of the Study Project
Preparing the training data:
- ```parse_pubmed_data.py```: creates our document corpus from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) abstracts and saves it in a ```pmid2info.json``` 
- ```parse_pmc_data.py```: extracts query-article pairs from PMC and saves them in a ```qid2info.json``` and ```train.jsonl```
- ```utils.py```: contains some utilities to analyze our training data 
- ```create_training_data_iterative.sh```: example shell-script to generate the training data for our iterative approach

Training and evaluating our models:
- Code to [train the retriever models](https://github.com/duylethanh/MedCPT/tree/main/retriever)
- Code to [evaluate the retriever models](https://github.com/duylethanh/MedCPT/tree/main/evals)

