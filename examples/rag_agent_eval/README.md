# RAG Agent Eval

## Preprocess Dataset

```bash
python -m examples.rag_agent_eval.preprocess_dataset ~/Desktop/Oracle\ evals\ data/ISO_13485_2016/ISO_13485_2016_01_S007_00_GM_llama-v3p1-405b-instruct_EM_text-embedding-3-large_K_30_175.xlsx ./rag_preprocessed
```

## Score Dataset

```bash
python -m examples.rag_agent_eval.score_dataset localhost 5000 ./rag_preprocessed/processed_ISO_13485_2016_01_S007_00_GM_llama-v3p1-405b-instruct_EM_text-embedding-3-large_K_30_175.xlsx.csv
```

## RAG Agent Eval Generate and Score Dataset

```bash
python -m examples.rag_agent_eval.rag_agent_generate localhost 5000 ./rag_preprocessed/processed_ISO_13485_2016_01_S007_00_GM_llama-v3p1-405b-instruct_EM_text-embedding-3-large_K_30_175.xlsx.csv
```

```bash
python -m examples.rag_agent_eval.score_dataset localhost 5000 ./rag_preprocessed/processed_ISO_13485_2016_01_S007_00_GM_llama-v3p1-405b-instruct_EM_text-embedding-3-large_K_30_175.xlsx.csv
```