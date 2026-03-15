# GeoLLM-Report-Interpreter
## GeoLLM: Automated Interpretation of Geotechnical Reports and Boring Logs Using Large Language Models

This project proposes GeoLLM, a domain-specialized document intelligence system that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to automatically extract, normalize, and structure engineering-relevant information from geotechnical reports. Unlike generic document AI systems, GeoLLM incorporates geotechnical semantics, engineering constraints, and schema-driven extraction to ensure outputs suitable for downstream engineering workflows such as preliminary foundation assessment and design verification.

The system ingests raw geotechnical documents (PDFs, scanned boring logs, and laboratory reports), performs robust text extraction and chunking, embeds domain-specific content into a vector database, and queries an LLM using carefully engineered prompts to generate validated, structured outputs (JSON/Excel). The architecture is modular, auditable, and deployable, supporting both academic experimentation and practical engineering use.

From a research perspective, the project evaluates LLM effectiveness in technical civil engineering language, compares prompt-based and retrieval-augmented extraction strategies, analyzes error modes, and assesses consistency against human expert interpretations. The long-term vision is a geotechnical AI co-pilot that enhances standardization, scalability, and efficiency in subsurface data interpretation.


```
GeoLLM-Report-Interpreter/
├── data/                    # Raw & processed datasets (do NOT commit large PDFs; use .gitignore)
│   ├── raw/                 # Original geotechnical reports, boring logs (add samples only)
│   ├── processed/           # Cleaned text/JSON outputs (gitignored if large)
│   └── synthetic/           # AI-generated reports for augmentation
├── notebooks/               # Exploratory Jupyter notebooks ( EDA, prototyping, experiments)
│   ├── 01_data_exploration.ipynb
│   ├── 02_prompt_engineering.ipynb
│   └── 03_evaluation.ipynb
├── src/                     # Core reusable Python code (modular)
│   ├── __init__.py
│   ├── document_processing.py   # PDF loading, OCR, text extraction, chunking
│   ├── embedding.py             # Embedding generation & vector store
│   ├── retrieval.py             # Retrieval logic
│   ├── llm_chain.py             # Prompt templates, LLM calls, output parsing
│   ├── schemas.py               # Pydantic/JSON schemas for structured output
│   └── utils.py                 # Helpers (logging, config, etc.)
├── app/                     # Streamlit web application (later phase)
│   ├── main.py
│   └── components.py
├── tests/                   # Unit & integration tests (add gradually)
├── configs/                 # Configuration files
│   └── config.yaml          # API keys (never commit), model settings, chunk sizes
├── prompts/                 # Text files or YAML with domain-specific prompts
│   └── extraction_prompts.yaml
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignore data/, __pycache__/, .env, large files
├── README.md                # Project overview, installation, usage, architecture diagram
├── LICENSE                  # e.g., MIT
├── pyproject.toml           # Optional: if using poetry/ modern packaging
└── docs/                    # Documentation, literature notes, meeting logs
    └── literature_review.md

```
15/03/2026



                ┌──────────────────────────────┐
                │        User Applications     │
                │  (Chatbot / API / UI)       │
                └──────────────┬──────────────┘
                               │
                        API Gateway
                               │
                ┌──────────────▼──────────────┐
                │     LLM Orchestration Layer │
                │  Prompt templates / Agents  │
                └──────────────┬──────────────┘
                               │
               ┌───────────────▼────────────────┐
               │  Retrieval Layer (RAG System)  │
               │  Embeddings + Vector Database  │
               └───────────────┬────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │   LLM Inference   │
                     │   Model Serving   │
                     └─────────┬─────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Model Training / Tuning   │
                │ (LoRA, RLHF, PEFT, etc.)    │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │ Data Pipeline & Processing  │
                │  ingestion → cleaning       │
                └──────────────┬──────────────┘
                               │
                    Data Lake / Storage
