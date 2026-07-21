# RAG-Based Resume Profile Matching

A retrieval-augmented generation (RAG) system that intelligently matches job descriptions to the most relevant candidate resumes using semantic search and hybrid BM25 ranking.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── resume_rag.py          # Document processing, embedding, ChromaDB indexing
├── job_matcher.py         # Hybrid search engine and scoring
├── rag_analysis.ipynb     # Experimentation notebook with visualizations
├── resumes/               # 42 candidate resume files (.txt)
├── job_descriptions/      # 6 job description files (.txt)
└── requirements.txt
```

## How It Works

### Part A — RAG System (`resume_rag.py`)
- Loads resumes from the `resumes/` directory (`.txt` and `.pdf` supported)
- Chunks documents by section — Education, Experience, Skills, Projects, etc.
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings + metadata (name, skills, experience years, education) in ChromaDB

### Part B — Job Matching Engine (`job_matcher.py`)
- Accepts a job description as input
- Runs **semantic search** (cosine similarity via ChromaDB) and **BM25 keyword search** in parallel
- Fuses scores with configurable weights (default: 60% semantic, 40% BM25)
- Returns top-K matches scored 0–100 with reasoning and matched skills

## Usage

**Index all resumes:**
```bash
python resume_rag.py
```

**Match a job description (top 10 results):**
```bash
python job_matcher.py job_descriptions/jd_senior_ml_engineer.txt 10
```

**Run against all job descriptions at once:**
```bash
python job_matcher.py
```

Results are saved to `match_results.json` and `all_match_results.json`.

## Output Format

```json
{
  "job_description": "...",
  "retrieval_latency_seconds": 0.42,
  "top_matches": [
    {
      "candidate_name": "Alex Morgan",
      "resume_path": "resumes/alex_morgan.txt",
      "match_score": 91.5,
      "matched_skills": ["Python", "PyTorch", "MLflow", "Docker"],
      "relevant_excerpts": ["..."],
      "reasoning": "Candidate scored 91.5/100. Strong alignment in [experience] section..."
    }
  ]
}
```

## Notebook

Open `rag_analysis.ipynb` in Jupyter for:
- Indexed collection stats
- Single JD match demo
- Batch evaluation across all 6 JDs
- Score distribution histogram and candidate heatmap
- Hybrid weight tuning sweep
- Performance metrics (latency, accuracy)

```bash
jupyter notebook rag_analysis.ipynb
```

## Dataset

- **42 resumes** covering: ML Engineers, Data Scientists, DevOps/SRE, Full Stack Developers, Cybersecurity Analysts, Data Engineers, NLP Researchers, Cloud Architects, and more
- **6 job descriptions**: Senior ML Engineer, Data Engineer, DevOps/SRE, Full Stack Developer, Cybersecurity Analyst, Python Backend Developer

## Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Resume and JD embeddings |
| `chromadb` | Persistent vector database |
| `rank-bm25` | Keyword-based BM25 search |
| `pandas / numpy` | Data processing |
| `matplotlib / seaborn` | Visualizations |
| `pypdf` | PDF resume support |

