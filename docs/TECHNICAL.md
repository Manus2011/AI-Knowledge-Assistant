# Technical documentation

## Request flow: uploading a document

1. `POST /documents/upload` (needs a token). Empty files get a 400.
2. File is saved under `data/` (filename stripped of any path parts, no overwriting).
3. `parser_service` extracts text (txt, pdf via pypdf).
4. Categorizer: TF-IDF + logistic regression (or MLP with `CATEGORIZER=mlp`). If the top probability is under 0.35 the label is `other`.
5. Keywords: TF-IDF against the user's other docs once there are at least 3, otherwise a built-in fallback corpus.
6. Text is split into ~500 character chunks on sentence boundaries with 80 characters of overlap.
7. Each chunk is embedded and stored in sqlite (`chunks` table, float32 blob).

## Request flow: asking a question (`POST /chat/documents`)

1. Embed the question, cosine similarity against the user's chunks (numpy dot product, vectors are L2 normalized).
2. Keep the top 4 over a minimum score of 0.08. Nothing above it means "couldn't find anything" without calling the model.
3. Build a prompt with the numbered chunks as context and send it with the last 10 messages of the conversation.
4. Return the answer plus sources (document, chunk, score).

## Agent (`POST /agent/run`)

LangGraph graph: `route` -> one of `search | list | stats | chat | delete`.
The router asks the model for JSON when an API key exists, otherwise uses regex rules.
`delete` finds the document by filename, then the `approve` node calls `interrupt()`,
which pauses the run. `POST /agent/approve` resumes it with true/false. Threads are namespaced per user.

## LangChain chain (`POST /chain/ask`)

`ChatPromptTemplate | RunnableLambda(llm) | StrOutputParser`, wrapped in `RunnableWithMessageHistory` with a sqlite backed history class, so the memory is the same `messages` table the other chat routes use.

## Deep learning (week 4)

`neural_categorizer.py` is an LSTM (embedding, LSTM, linear) trained at startup when `CATEGORIZER=lstm`. If torch isn't installed the app falls back to the sklearn model. `exercises/week4_dl/` has the CNN, the numpy network and the text classifier comparisons.

## Data model (sqlite)

- `users(username, password_hash, salt)`: pbkdf2-sha256, 200k iterations, per-user salt
- `documents(id, filename, ..., extracted_text, category, keywords, owner)`
- `chunks(doc_id, idx, text, embedding)`
- `messages(conversation_id, role, content)`: conversation ids are prefixed with the username

## Security notes

- Other users' documents return 404, not 403
- Login failure message is the same for unknown user and wrong password
- Unhandled errors are logged server side and the client only sees a generic 500
- Default JWT secret logs a warning on startup

## Testing

`python -m pytest`: 53 tests covering upload, persistence, categorizer backends, chat memory, chunking,
retrieval, RAG, agent routing and approval flow, auth, user isolation, error handler.
The Anthropic provider is tested with a fake client, no network calls.

## Known gaps

See the limitations section in the README.
