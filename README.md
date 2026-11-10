# HR Policy Assistant

An interactive HR policy question-answering application built with Streamlit, LangChain, Google Gemini, and FAISS. The assistant retrieves relevant company policies and can answer employee-specific leave and PTO questions using the local HRIS sample data.

## Features

- Semantic search over company policies and employee records with FAISS.
- Gemini-powered answers using a retrieval-augmented generation (RAG) chain.
- Filters inactive policies before sending context to the language model.
- Employee-specific PTO and leave balance lookups by employee ID.
- Protects employee data by requiring an employee ID for personal questions and instructing the model not to reveal other employees' information.
- Duplicate-result filtering for policy and employee documents.
- Automated unit tests for document loading, retrieval, vector-store behavior, and HRIS validation.

## Architecture

1. `DocumentLoader` reads `data/policies.json` and `data/hris_api.json` and converts each record into a LangChain `Document`.
2. `PolicyVectorStore` creates or loads a local FAISS index using Gemini embeddings.
3. `HRPolicyRAG` retrieves relevant documents, keeps only active policies, adds authorized employee HRIS data when needed, and generates an answer with Gemini.
4. `app.py` provides the Streamlit chat interface and caches the loaded RAG components for the session.

The sample data contains 10 policies and 6 employee records. The 2024 annual leave policy is marked inactive and is excluded from answer context.

## Requirements

- Python 3.13 or newer
- A Google AI API key with access to Gemini chat and embedding models
- `uv` (recommended) or `pip`

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/AliRizvi433/HandsonPYQ.git
cd HandsonPYQ

uv sync
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_api_key
```

## Build the FAISS index

The Streamlit application expects a generated `faiss_index/` directory. Build it from the bundled policy and HRIS data:

```bash
uv run python test_retrieval.py
```

With an activated virtual environment, use:

```bash
python test_retrieval.py
```

The generated index is local and is ignored by Git.

## Run the application

After creating the index, start Streamlit:

```bash
uv run streamlit run app.py
```

Open the displayed local URL in a browser. Enter an employee ID, such as `EMP101`, before asking personal questions about PTO or leave balances. General policy questions do not require an employee ID.

## Example questions

- `How many sick leave days do employees get?`
- `Can employees work from home?`
- `What is my PTO balance?` with employee ID `EMP101`
- `How many vacation days do I have left?` with employee ID `EMP103`

## Run tests

Run the unit-test suite with:

```bash
uv run pytest
```

Some integration scripts require a generated FAISS index and a configured `GOOGLE_API_KEY`:

```bash
uv run python test_full_rag.py
uv run python test_gemini.py
```

## Project structure

```text
.
├── app.py                         # Streamlit user interface
├── data/
│   ├── hris_api.json              # Sample employee and PTO records
│   └── policies.json              # Sample active and inactive policies
├── src/pyqproj/
│   ├── document_loader.py         # JSON-to-LangChain document conversion
│   ├── rag_chain.py               # Retrieval and Gemini answer generation
│   └── vector_store.py            # FAISS creation, persistence, and search
├── test_retrieval.py              # Builds and exercises the FAISS index
├── test_full_rag.py               # End-to-end RAG examples
├── test_gemini.py                 # Gemini connectivity checks
└── tests/                         # Unit tests
```

## Data and privacy note

The JSON files contain demonstration data only. Do not place real employee records or API keys in the repository. Keep `.env` private, and replace the sample HRIS source with an access-controlled service before using this pattern with production data.
