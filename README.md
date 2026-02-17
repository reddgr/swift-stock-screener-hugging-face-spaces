---
title: Swift Stock Screener
emoji: 💹
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Search over 12k companies by theme (VSS) or similarity (NNS)
---

## Overview

**Swift Stock Screener (SSS)** is an interactive application that enables semantic and similarity-based stock screening across ~13,000 equities.  
It combines **semantic vector search** on company descriptions with **nearest neighbor search** on financial fundamentals.  

The app allows two main modes:
- **Theme search**: find stocks relevant to a concept (e.g., *“lithium”*, *“cloud computing”*).
- **Ticker similarity search**: find peers to a given stock (e.g., *NVDA* → other semiconductors).

Data (fundamentals and descriptions) is sourced from **Yahoo Finance**. Company **descriptions used in the semantic search** come directly from Yahoo Finance long summaries.

---

## Features

- **Semantic Theme Search**  
  Uses `FinLang/finance-embeddings-investopedia` to embed queries and company descriptions.  
  DuckDB with **HNSW vector index** executes approximate nearest neighbor search.  
  Re-ranking adds:
  - **Literal rewards** (companies explicitly mentioning query terms boosted).
  - **Brevity penalties** (companies with very short descriptions are penalized).

- **Ticker Similarity Search**  
  Uses normalized numeric fundamentals (e.g., Market Cap, P/E, Revenue Growth, Volatility).  
  Scikit-learn **NearestNeighbors** finds the closest peers.  
  A **Similarity Index (0–100)** is computed with a gamma distribution scaling to give intuitive scores.

- **Interactive Results Table**  
  Paginated grid view with columns:
  - Name, Country, Sector, Market Cap
  - 1yr Return, Volatility, P/E, Revenue Growth, Dividend Yield, Beta  
  Negative values are highlighted red.  

- **Dynamic Filtering**  
  Click on a sector, industry, or country in the grid to filter results interactively.  
  Multiple filters can be applied in succession.

- **Company Details View**  
  - Company name + Yahoo Finance description.  
  - Full fundamentals table.  
  - **Radar chart** of 5 normalized metrics: Beta, Debt/Equity, 1yr Return, Revenue Growth, Volatility.

- **Controls**  
  - **Theme box**: free text search.  
  - **Ticker box**: find peers to a stock.  
  - **Search** / **Reset** / **Random ticker** buttons.  
  - Sorting by any metric.  

---

## Architecture

- **Data**: Yahoo Finance fundamentals + descriptions.  
- **Embeddings**: `FinLang/finance-embeddings-investopedia` (768-dim vectors).  
- **Vector DB**: DuckDB with VSS extension + HNSW cosine index.  
- **Ticker similarity**: Scikit-learn `NearestNeighbors` with Euclidean distance over normalized features.  
- **UI**: Python + Gradio Blocks.  
- **Visualization**: Plotly for radar chart.

---
