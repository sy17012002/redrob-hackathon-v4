# Redrob Hackathon v4 - Team [XYZ001]

This repository contains our Two-Stage Retrieval & Re-ranking pipeline for the Redrob AI Engineer matching hackathon.

## Architecture Overview
Our pipeline is designed for production-level speed and precision, completely bypassing the standard 2-hour CPU bottleneck associated with deep learning models. 
1. **Stage 1 (Lexical Screener):** Streams 100,000 records, filtering out non-technical honeypots via a strict profession blacklist and extracting the top 400 technical candidates using heuristic scoring (Experience, Location, Keyword Density).
2. **Stage 2 (Semantic Re-ranking):** Encodes the elite 400 profiles using a local `BAAI/bge-small-en-v1.5` transformer model, utilizing FAISS flat indexing to rank the candidates based on pure semantic similarity to the Job Description.


## Setup Instructions

1. Ensure you have Python 3.9+ installed in your environment.
2. Clone this repository and navigate to the root directory.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Execution Command

The ranking script requires explicitly declared flags for input and output pathways to ensure absolute sandbox test reproducibility. Execute the pipeline using the following single command format from the repository root:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
