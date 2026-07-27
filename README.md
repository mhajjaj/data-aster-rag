# Data Aster RAG System

社内AI担当として、共有ドライブに蓄積された案件資料から必要な情報を探し出し、社内から寄せられる質問に正確に回答するRAG（Retrieval-Augmented Generation）システムです。

本システムは単なるテキスト検索を超え、画像やグラフ、表などの非構造化・構造化データを統合し、複数資料や複数案件にまたがる情報を検索・照合・集計して、根拠に基づいた回答を生成します。

さらに、未知の案件や資料が追加されても同じ処理方針で対応できる、**自律型RAG（Agentic RAG）パイプライン**を目指します。

## Architecture

```
Raw Documents -> Ingestion -> Indexing -> Retrieval -> Generation -> Answer
                                      ^                    ^
                                       \                  /
                                        \                /
                                         Agentic Orchestrator
```

## Stages

- Stage 1: Project scaffolding and document ingestion pipeline
- Stage 2: Multi-modal processing (text + image + table extraction)
- Stage 3: Vector indexing and embedding store
- Stage 4: Retrieval with query understanding
- Stage 5: Agentic RAG orchestration
- Stage 6: Answer generation with citation
- Stage 7: Evaluation and submission pipeline

## Directory Structure

```
data-aster-rag/
├── config/            # Configuration files
├── data/              # Raw and processed data (not committed)
├── src/               # Source code
│   ├── ingestion/     # Document loading & parsing
│   ├── indexing/      # Embeddings & vector store
│   ├── retrieval/     # Search & query refinement
│   ├── generation/    # LLM answer generation
│   ├── agent/         # Agentic RAG orchestrator
│   └── utils/         # Helpers
├── scripts/           # Executable scripts
├── tests/             # Unit tests
└── notebooks/         # Experiment notebooks
```

## Setup

```bash
pip install -r requirements.txt
```

## License

For SIGNATE Competition use only.
