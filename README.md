# Enterprise AI Knowledge Assistant

Capstone project for the Onshore Internship Program (8 weeks).
Upload documents, have them categorized and indexed, then ask questions about them
(RAG with sources). There's also an agent that can use tools and asks for approval before deleting anything.

![architecture](docs/architecture.png)

## Run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # set JWT_SECRET, optionally ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`. Register at `/auth/register`, log in at `/auth/login`,
then click Authorize and paste the token.

Tests: `python -m pytest` (53 tests, the 4 deep learning ones need `pip install -r requirements-dl.txt`). Docker: see [DEPLOYMENT.md](DEPLOYMENT.md).

## What it does

- Upload .txt / .pdf, text extraction, auto category (resume / meeting_notes / report / contract / other), TF-IDF keywords
- Chunking + embeddings + vector search over your uploaded docs
- `/chat` plain chat with conversation memory, `/chat/documents` answers from your docs with sources
- `/chain/ask` same document chat built as a LangChain chain (prompt | llm | parser) with message history in sqlite
- `/agent/run` LangGraph agent: search docs, list docs, stats, delete (needs approval via `/agent/approve`)
- JWT login, every user only sees their own documents
- Request logging, gzip, timing header, query embedding cache, error handling, Swagger docs, Docker

More detail in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Things to know (limitations)

- Without `ANTHROPIC_API_KEY` the answers are extractive: it picks the best matching sentences from your docs. With a key it uses Claude to write the answer from the retrieved chunks (checked by hand end to end). The automated tests use a fake LLM so they don't need a key or network
- Embeddings are hashed words/bigrams with crude stemming. They match on shared wording, not meaning, so synonyms won't match. A real embedding model would be the next upgrade (only `embedding_service.embed()` changes)
- The vector store loads a user's chunks and does one numpy matrix product. Fine for hundreds of docs, not for millions
- Classifier training data is small and partly synthetic. 80% on my own held-out test set does not mean 80% on real company documents
- The LSTM classifier is at chance (25%) on phrasing it never saw, because most of those words aren't in its vocabulary. It does fine on text like its training data. The CNN is solid on digits
- Dockerfile and compose file are included but I haven't tested them
- Agent state uses an in-memory checkpointer, pending approvals are lost on restart

## Numbers I measured

- Document categorizer: 80% accuracy on 50 held-out self-written docs
- Retrieval on 5 policy docs and 15 questions I wrote (`python scripts/eval_retrieval.py`): correct doc ranked 1st for 12/15, in the top 3 for 15/15
- Digit recognizer (numpy neural net from scratch): 96.9% on the sklearn digits test split
- Week 4 text classifier on unseen phrasing: logreg 54%, MLP 55%, LSTM 25% (chance for 4 classes). The neural models don't beat logreg on this tiny synthetic data
- CNN on the 8x8 digits test split: 97.3% (numpy net from scratch 96.9%)
- Embedding experiments (`python scripts/embedding_experiments.py`): on my 15 questions, 100 character chunks with 4096 dims got 14/15 first-place hits vs 12/15 for 500 character chunks, and tiny 64 dim vectors only got 7 to 8 because of hash collisions
- Search latency over 200 chunks: about 1.5 ms median (`python scripts/benchmark_api.py`)

## Progress Log

- **Week 1** — Repo structure, Git workflow, FastAPI skeleton, document upload + parsing (txt/pdf), pandas-based stats endpoint
- **Week 2** — ML exercises (spam classifier, customer prediction, data viz), document categorization wired into the app
- **Week 3** — NLP exercises (resume classifier, sentiment analyzer, keyword extractor), automatic keyword extraction wired into document upload
- **Week 4** — Neural net from scratch, CNN (pytorch) on digits, MLP and LSTM text classifiers (`CATEGORIZER=mlp` or `lstm`), sqlite storage + pytest suite
- **Week 5** — LLM service (Anthropic + extractive fallback), prompt experiments, chat endpoint with conversation memory
- **Week 6** — Chunking, embeddings, numpy vector store, search, RAG chat with sources
- **Week 7** — LangGraph agent with tools and human approval for deletes, LangChain chain with memory
- **Week 8** — JWT auth, per-user data, logging, error handling, Swagger docs, Docker
