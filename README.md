# NewsBot Intelligence System 2.0

**ITAI 2373 - Final Project** | Advanced NLP Integration and Analysis Platform

An end-to-end news intelligence platform that classifies articles, tracks
sentiment, extracts entities, discovers topics, summarizes content, analyzes
coverage across languages, and answers natural-language questions about a
news corpus - built on top of the midterm NewsBot foundation.

## Team (Working individully for now. I am planning to add transformer later)

| Name | Role / 
|---|---|
| King Owusu | Building and Organizing |


See `docs/individual_contributions.md` for a detailed breakdown.

## What This System Does

| Module | Capability |
|---|---|
| **A - Advanced Content Analysis** | Multi-model classification with confidence scoring, LDA/NMF topic discovery, sentiment evolution tracking, entity relationship mapping |
| **B - Language Understanding & Generation** | Extractive summarization, semantic search + query expansion, automatic business-insight generation |
| **C - Multilingual Intelligence** | Offline language detection, translation integration, cross-lingual sentiment/coverage comparison |
| **D - Conversational Interface** | Natural-language query understanding with follow-up context, e.g. *"Show me positive tech news from this week"* |

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet vader_lexicon averaged_perceptron_tagger

# Run tests
pytest tests/ -v

# Explore the notebooks (Cells executed, open the notebooks directory to view)
jupyter notebook notebooks/

# Run the Django web app
python manage.py migrate
python manage.py runserver
# visit http://127.0.0.1:8000/
```

By default the system uses a small bundled offline sample dataset
(`data/sample/sample_news.csv`) so everything above runs immediately with
no external downloads. Alternatively, you can download the BBC News Train 
dataset on Kaggle and place it in `data/raw/` - it will be picked up automatically.

## Repository Structure

```
ITAI2373-NewsBot-Final/
├── README.md
├── requirements.txt
├── config/                    # settings.py, api key template
├── src/                       # core NLP package (see docs/api_reference.md)
│   ├── data_processing/
│   ├── analysis/
│   ├── language_models/
│   ├── multilingual/
│   ├── conversation/
│   └── utils/
├── newsbot_web/                # Django project (settings, urls, wsgi)
├── newsbot_app/                 # Django app (views, models, templates wiring, API)
├── templates/                  # Django HTML templates (Bootstrap-based UI)
├── notebooks/                  # 01-07, executed with real outputs
├── tests/                      # pytest suite (21 tests)
├── data/
│   ├── raw/                    # place BBC News Train.csv here (not committed)
│   ├── sample/                 # bundled offline demo dataset
│   ├── processed/              # notebook-generated intermediate files
│   ├── models/
│   └── results/
├── docs/                       # technical, user, API, deployment docs
└── reports/                    # executive summary, technical report, slides
```

## Web Application (Bonus)

Built with **Django** as I am more familiar with and per project requirements. instead of Flask, with a
Bootstrap-based dashboard, single-article analysis, a conversational query
interface, batch CSV processing, and a JSON API (`/api/analyze/`,
`/api/query/`) for external integrations. See `docs/api_reference.md` and
`docs/deployment_guide.md`.

## Testing

```bash
pytest tests/ -v --cov=src
```
21 tests covering preprocessing, classification, topic modeling, sentiment,
NER, the multilingual pipeline, and the conversational interface end-to-end.

## Academic Integrity Note

This project was developed with AI assistance (Claude) for code scaffolding,
documentation drafting, and the Django web application, building on the
completed midterm NewsBot analysis. See `reports/reflective_journal.md` for details on
how AI assistance was used, per the course's academic integrity
requirements.

## License

Educational project for ITAI 2373. Not licensed for commercial use.
