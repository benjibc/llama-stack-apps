# RAG Agent Eval

## 0. Preprocess Dataset

```bash
python -m examples.rag_agent_eval.preprocess_dataset <path to raw unprocessed dataset> <path to output directory>
```

## 0.5 (Optional) RAG Agent Eval Generate

```bash
python -m examples.rag_agent_eval.rag_agent_generate localhost 5000 <path to docs dir> <path to input query dataset>
```

## 1. Score Dataset

```bash
python -m examples.rag_agent_eval.score_dataset localhost 5000 <path to processed dataset>
```