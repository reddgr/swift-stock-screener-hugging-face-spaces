'''
Swift Stock Screener (SSS)
Copyright 2025 David González Romero

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

App URL: https://huggingface.co/spaces/reddgr/sss
'''

### DEBUGGING COMMAND (DGR):
# cd C:\Users\david\Documents\git\sss; python app.py
from pathlib import Path
from typing import Tuple
import pandas as pd
import gradio as gr
import json
from datetime import datetime

import duckdb
from sentence_transformers import SentenceTransformer

USE_DOTENV = False

ROOT = Path(__file__).parent

JSON_PATH = ROOT / "json"
EMB_MODEL_PATH = "FinLang/finance-embeddings-investopedia" # Hugging Face Hub embeddings model name
DOTENV_PATH = ROOT.parent.parent / "apis" / ".env"
PARQUET_PATH = ROOT / "parquet" / "app_dataset.parquet"
# DUCKDB_PATH = ROOT / "db" / "sss_vectordb.duckdb"

from src import front_dataset_handler as fdh, app_utils as utils, semantic_search as ss, env_options
tokens = env_options.check_env(use_dotenv=USE_DOTENV, dotenv_path=DOTENV_PATH, env_tokens = ["HF_TOKEN"])

emb_model = SentenceTransformer(EMB_MODEL_PATH, token = tokens.get("HF_TOKEN"))

### Comprobación de la fecha de actualización del dataset Parquet ###
def get_parquet_last_update(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    except FileNotFoundError:
        return "unavailable"

PARQUET_LAST_UPDATE = get_parquet_last_update(PARQUET_PATH)
print(f"Parquet dataset last update: {PARQUET_LAST_UPDATE}")

#### CONEXIÓN DE DUCKDB CON EL DATASET PARA INDEXAR ####
print("Initializing DuckDB connection...")
con = duckdb.connect()

create_table_query = f"""
        INSTALL vss;
        LOAD vss;
        SET hnsw_enable_experimental_persistence = true;
        CREATE TABLE vector_table AS 
        SELECT *, embeddings::float[{emb_model.get_sentence_embedding_dimension()}] as embeddings_float 
        FROM '{PARQUET_PATH}';
        """

con.sql(create_table_query)

print("Indexing data for vector search...")
create_index_query = f"""
        CREATE INDEX sss_hnsw_index ON vector_table USING HNSW (embeddings_float) WITH (metric = 'cosine');
        """
con.sql(create_index_query)

# ESTADO GLOBAL
last_result_df: pd.DataFrame = pd.DataFrame()
last_search_type: str = ""
last_search_query: str = ""
last_column_filters: list[tuple[str, str]] = []
last_sort_col_label: str = ""
last_sort_dir: str = ""
selected_ticker: str = ""

# ---------------------------------------------------------------------------
# CONFIG --------------------------------------------------------------------
# ---------------------------------------------------------------------------

app_dataset = con.execute(
    "SELECT * FROM read_parquet(?)",
    [str(PARQUET_PATH)],
).fetchdf()
dh_app = fdh.FrontDatasetHandler(app_dataset=app_dataset)
maestro = dh_app.app_dataset[dh_app.app_dataset['quoteType']=='EQUITY'].copy()
print("maestro_columns", maestro.columns.to_list())
maestro_etf = dh_app.app_dataset[dh_app.app_dataset['quoteType']=='ETF'].copy()

with open(JSON_PATH / "app_column_config.json", "r") as f:
    variables_busq_norm = json.load(f)["variables_busq_norm"]

with open(JSON_PATH / "app_column_config.json", "r") as f:
    caracteristicas = json.load(f)["cols_tabla_equity"]

with open(JSON_PATH / "app_column_config.json", "r") as f:
    caracteristicas_etf = json.load(f)["cols_tabla_etfs"]

with open(JSON_PATH / "app_column_config.json", "r") as f:
    company_details_cols = json.load(f)["company_details_cols"]

with open(JSON_PATH / "cat_cols.json", "r") as f:
    cat_cols = json.load(f)["cat_cols"]

with open(JSON_PATH / "col_names_map.json", "r") as f:
    rename_columns = json.load(f)["col_names_map"]

with open(JSON_PATH / "gamma_params.json", "r") as f:
    gamma_params = json.load(f)

with open(JSON_PATH / "semantic_search_params.json", "r") as f:
    semantic_search_params = json.load(f)["semantic_search_params"]

# Columnas a estilizar en rojo si son negativas
neg_display_cols = [rename_columns.get(c, c)
                    for c in ("ret_365", "revenueGrowth")]

# Parámetros de la función de distribución de distancias:
shape, loc, scale = gamma_params["shape"], gamma_params["loc"], gamma_params["scale"]
max_dist, precision_cdf = gamma_params["max_dist"], gamma_params["precision_cdf"]
y_cdf, _ = dh_app.configura_distr_prob(shape, loc, scale, max_dist, precision_cdf)

# Parámetros de búsqueda VSS:
k = semantic_search_params["k"]
brevity_penalty = semantic_search_params["brevity_penalty"]
min_length = semantic_search_params["min_length"]
reward_for_literal = semantic_search_params["reward_for_literal"]
first_term_reward = semantic_search_params["first_term_reward"]
partial_match_factor = semantic_search_params["partial_match_factor"]
print(f"VSS params: k={k}, brevity_penalty={brevity_penalty}, reward_for_literal={reward_for_literal}, partial_match_factor={partial_match_factor}", end="")
print(f", min_length={min_length}, first_term_reward={first_term_reward}")

filtros_keys = caracteristicas[2:]

MAX_ROWS = 13000
ROWS_PER_PAGE = 100

# ---------------------------------------------------------------------------
# FUNCIONES UI --------------------------------------------------------------
# ---------------------------------------------------------------------------

# Dejamos en este módulo (en lugar de app_utils) funciones específicas de gestión de la interfaz

def _paginate(df: pd.DataFrame, page: int, per_page: int = ROWS_PER_PAGE) -> Tuple[pd.DataFrame, str]:
    total_pages = max(1, (len(df) + per_page - 1) // per_page)
    page        = max(1, min(page, total_pages))
    slice_df    = df.iloc[(page-1)*per_page : (page-1)*per_page + per_page]
    slice_df = utils.styler_negative_red(slice_df, cols=neg_display_cols)
    return slice_df, f"Page {page} of {total_pages}"


def search_dynamic(ticker: str, page: int, *filtros_values) -> Tuple[pd.DataFrame, str]:
    global last_result_df

    ticker = ticker.upper().strip()
    if ticker == "":
        last_result_df = pd.DataFrame()
        return pd.DataFrame(), "Page 1 of 1"

    filtros = dict(zip(filtros_keys, filtros_values))

    neighbors_df = dh_app.vecinos_cercanos(
        df=maestro,
        variables_busq=variables_busq_norm,
        caracteristicas=caracteristicas,
        target_ticker=ticker,
        y_cdf=y_cdf,
        precision_cdf=precision_cdf,
        max_dist=max_dist,
        n_neighbors=len(maestro),
        filtros=filtros,
    )

    if isinstance(neighbors_df, str):
        last_result_df = pd.DataFrame()
        return pd.DataFrame(), "Page 1 de 1"

    neighbors_df.reset_index(inplace=True)
    neighbors_df.drop(columns=["distance"], inplace=True)
    # neighbors_df = format_results(neighbors_df)
    neighbors_df = utils.format_results(neighbors_df, rename_columns)

    last_result_df = neighbors_df.head(MAX_ROWS).copy()
    return _paginate(last_result_df, page)


def search_theme(theme: str, page: int, *filtros_values) -> Tuple[pd.DataFrame, str]:
    global last_result_df
    query = theme.strip()
    if query == "":
        last_result_df = pd.DataFrame()
        return pd.DataFrame(), "Page 1 of 1"

    # Llamada al algoritmo de búsqueda, que devuelve un dataframe con k activos:
    result_df = ss.duckdb_vss_local(
        model=emb_model,
        duckdb_connection=con,
        query=query,
        k=k,
        brevity_penalty=brevity_penalty,
        min_length = min_length,
        reward_for_literal=reward_for_literal,
        first_term_reward=first_term_reward,
        partial_match_factor=partial_match_factor,
        table_name="vector_table",
        embedding_column="embeddings"
    )
    theme_dist = result_df[['ticker', 'distance']].rename(columns={'distance': 'Search dist.'})
    # Cruzamos el dataframe de distancias con el maestro y mantenemos las columnas originales: 
    clean_feats = [c for c in caracteristicas if c != 'ticker']
    # indexamos por ticker para cruzar las tablas:
    maestro_subset = maestro.set_index('ticker')[clean_feats]
    merged = theme_dist.set_index('ticker').join(maestro_subset, how='inner').reset_index()
    # Reordenamos las columnas y añadimos la distancia:
    ordered_cols = ['ticker'] + clean_feats + ['Search dist.']
    merged = merged[ordered_cols]
    # Ajustamos los formatos de las columnas:
    formatted = utils.format_results(merged, rename_columns)
    last_result_df = formatted.head(MAX_ROWS).copy()
    return _paginate(last_result_df, page)


def _compose_summary() -> str:
    parts = []
    if last_search_type == "theme":
        parts.append(f"Theme search for '{last_search_query}'")
    elif last_search_type == "ticker":
        parts.append(f"Ticker search for '{last_search_query}'")
    if last_column_filters:
        fstr = ", ".join(f"{col} = '{val}'" for col, val in last_column_filters)
        parts.append(f"Filters: {fstr}")
    if last_sort_col_label:
        parts.append(f"Sorted by: {last_sort_col_label} ({last_sort_dir})")
    return ". ".join(parts)

def search_all(theme: str, ticker: str, page: int) -> tuple[pd.DataFrame,str,str,str,str]:
    global last_search_type, last_search_query, last_column_filters
    last_column_filters.clear()

    if theme.strip():
        last_search_type, last_search_query = "theme", theme.strip()
        df, label = search_theme(theme, page)
        # new_ticker, new_theme = "", theme.strip()
        new_ticker, new_theme = "", "" # limpia las cajas de búsqueda

    elif ticker.strip():
        last_search_type, last_search_query = "ticker", ticker.strip().upper()
        df, label = search_dynamic(ticker, page)
        # new_ticker, new_theme = last_search_query, ""
        new_ticker, new_theme = "", ""

    else:
        df, label = _paginate(last_result_df, page)
        new_ticker, new_theme = "", ""

    summary = _compose_summary()
    return df, label, new_ticker, new_theme, summary

def page_change(theme: str, ticker: str, page: int) -> tuple[pd.DataFrame,str,str,str,str]:
    return search_all(theme, ticker, page)


# ---------------------------------------------------------------------------
# SORTING -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def apply_sort(col_label: str, direction: str) -> tuple[pd.DataFrame, str, int, str]:
    global last_sort_col_label, last_sort_dir, last_search_type, last_search_query, last_column_filters, last_result_df

    # record selection and clear previous state
    last_sort_col_label, last_sort_dir = col_label or "", direction or ""
    last_search_type = last_search_query = ""
    last_column_filters.clear()

    # reload raw data
    df_raw = maestro[caracteristicas].head(MAX_ROWS).copy()

    # sort on original data column if specified
    if col_label:
        # reverse lookup original column key
        inv_map = {v: k for k, v in rename_columns.items()}
        orig_col = inv_map.get(col_label, col_label)
        asc = (direction == "Ascending")
        df_raw = df_raw.sort_values(
            by=orig_col,
            ascending=asc,
            na_position='last'
        ).reset_index(drop=True)

    # apply existing formatting helpers
    df_formatted = utils.format_results(df_raw, rename_columns)

    # update global and paginate
    last_result_df = df_formatted.copy()
    slice_df, label = _paginate(last_result_df, 1)
    summary = f"Sorted by: {col_label} ({direction})" if col_label else ""
    return slice_df, label, 1, summary



def reset_initial() -> tuple[pd.DataFrame,str,int,str,str,str]:
    global last_search_type, last_search_query, last_column_filters, last_sort_col_label, last_sort_dir, last_result_df
    last_search_type = last_search_query = ""
    last_column_filters.clear()
    last_sort_col_label = last_sort_dir = ""
    last_result_df = utils.format_results(maestro[caracteristicas].head(MAX_ROWS).copy(), rename_columns)
    slice_df, label = _paginate(last_result_df, 1)
    default_sort = rename_columns.get("marketCap","marketCap")
    return slice_df, label, 1, "", "", default_sort, ""


# ---------------------------------------------------------------------------
# DATOS INICIALES -----------------------------------------------------------
# ---------------------------------------------------------------------------

#last_result_df = utils.format_results(maestro[caracteristicas].head(MAX_ROWS).copy(), rename_columns)
#_initial_slice, _initial_label = _paginate(last_result_df, 1)

last_result_df = utils.format_results(maestro[caracteristicas].head(MAX_ROWS).copy(), rename_columns)
_initial_slice, _initial_label = _paginate(last_result_df, 1)
# Ticker por defecto es el primero de la lista
if not last_result_df.empty:
    selected_ticker = last_result_df.iloc[0][rename_columns.get('ticker','ticker')]
# Fetch initial company info
if selected_ticker:
    maestro_details = maestro[company_details_cols].copy()
    init_name, init_summary, init_details = utils.get_company_info(maestro_details, selected_ticker, rename_columns)
else:
    init_name, init_summary, init_details = "", "", pd.DataFrame()

# ---------------------------------------------------------------------------
# UI ------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _load_html(name: str) -> str:
    return (ROOT / "html" / name).read_text(encoding="utf-8")

html_front_layout = _load_html("front_layout.html")

with gr.Blocks(title="Swift Stock Screener, by Reddgr") as front:
    gr.HTML(html_front_layout)

    # ---------------------- TOP INPUT -------------------------------------
    with gr.Row(equal_height=True):
        theme_input = gr.Textbox(show_label=False, placeholder="Search a theme. i.e. 'lithium'", scale=2)
        ticker_input = gr.Textbox(show_label=False, placeholder="Enter a ticker symbol. i.e. 'nvda'", scale=1)   
        buscar_button = gr.Button("Search")
        gr.HTML("<div></div>")
        reset_button = gr.Button("Reset", elem_classes="small-btn")
        # gr.HTML("<div></div>")
        random_button = gr.Button("Random ticker", elem_classes="small-btn")

    # ---------------------- SEARCH SUMMARY ------------------------
    summary_display = gr.Markdown("", elem_classes="search-spec")

    # ---------------------- RESULTS ↔ COMPANY TABS ----------------------------
    with gr.Tabs(selected=0) as main_tabs:                  # 0 = “Results”
        # ---- TAB 1: GRID --------------------------------------------------
        with gr.TabItem("Grid"):
            output_df = gr.Dataframe(
                value=_initial_slice,
                interactive=False,
                elem_classes="df-cells",
            )

            with gr.Row():
                btn_prev  = gr.Button("Previous", elem_classes="small-btn")
                pagination_label = gr.Markdown(_initial_label)
                btn_next  = gr.Button("Next", elem_classes="small-btn")
                gr.Markdown("&nbsp;" * 20)
                sort_col = gr.Dropdown(
                    [rename_columns.get(c, c) for c in caracteristicas],
                    value=None,
                    label="Reset and sort by:",
                    allow_custom_value=False,
                    scale=2,
                )
                sort_dir = gr.Radio(
                    ["Ascending", "Descending"],
                    value="Descending",
                    label="",
                    scale=1,
                )

            # ---------------------- DATASET UPDATE DISPLAY ------------------------
            dataset_update_display = gr.Markdown(
                f"**Dataset last updated:** {PARQUET_LAST_UPDATE}",
                elem_classes="search-spec",
            )

        # ---- TAB 2: COMPANY --------------------------------------------------
        '''
        with gr.TabItem("Company details")as company_tab: ####
            company_title   = gr.Markdown(f"## {init_name}" if init_name else "### Company Name")
            company_summary = gr.Markdown(init_summary)
            company_details = gr.Dataframe(value=init_details, interactive=False)
        '''

        with gr.TabItem("Company details") as company_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    company_title = gr.Markdown(f"## {init_name}" if init_name else "### Company Name")
                    company_summary = gr.Markdown(init_summary)
                    company_details = gr.Dataframe(value=init_details, interactive=False)
                with gr.Column(scale=1):
                    company_chart_title = gr.Markdown("## Key Metrics Radar Chart")
                    company_plot = gr.Plot(visible=True)

    def on_company_tab():
        global selected_ticker
        print(f"DEBUG on_company_tab: selected_ticker={selected_ticker}")
        # if evt.selected and selected_ticker:
        if selected_ticker:
            maestro_details = maestro[company_details_cols].copy()
            # maestro_details.drop(columns=["embeddings"], inplace=True, errors="ignore")
            name, summary, details_df = utils.get_company_info(
                maestro_details, selected_ticker, rename_columns
            )
            
            # Create spider plot figure
            fig = None
            try:
                if not details_df.empty:
                    fig = utils.get_spider_plot_fig(details_df)
            except Exception as e:
                print(f"Error creating spider plot: {e}")          
            
            
            
            return (
                gr.update(value=f"## {name}"),
                gr.update(value=summary),
                gr.update(value=details_df),
                gr.update(value=fig),
                # gr.update()
                ### summary_display ##########
            )
        # No company selected – leave widgets as‑is
        return gr.update(), gr.update(), gr.update(), gr.update() # summary_display
    
    def _dbg_company_tab_select(*_):
        print("DEBUG company_tab.select event fired")
        return on_company_tab()

    company_tab.select(
        _dbg_company_tab_select,
        # on_company_tab,
        inputs=[],
        outputs=[company_title, company_summary, company_details, company_plot]
    )

    # ---------------------- TABLE SELECT (CLICK) ---------------------        
    page_state = gr.State(1)

    def on_table_select(evt: gr.SelectData):
        print(f"DEBUG on_table_select called: index={evt.index}, value={evt.value}")
        global last_result_df, selected_ticker
        row_i, col_i = evt.index
        if col_i == 0:
            ticker = evt.value
            print(f"DEBUG ticker extracted: {ticker}")
            selected_ticker = ticker
        elif col_i == 1 or (4 <= col_i <= 10):
            display_col = rename_columns.get("ticker", "ticker")
            ticker = last_result_df.iloc[row_i][display_col]
            print(f"DEBUG ticker extracted: {ticker}")
            selected_ticker = ticker      # keep global state in sync
        else:
            # Filter by column returns (df, pagination_label, page_number, summary)
            filtered_df, pagination, page, summary = filter_by_column(evt)
            # We need to return all 9 output values
            return (
                filtered_df,
                pagination,
                page,
                summary,
                gr.update(selected=0),  # Keep on the results tab
                gr.update(),  # company_title
                gr.update(),  # company_summary
                gr.update(),  # company_details
                gr.update()   # company_plot
            )
        
        maestro_details = maestro[company_details_cols].copy()
        name, summary, details_df = utils.get_company_info(maestro_details, ticker, rename_columns)

        # Create spider plot figure
        fig = None
        try:
            if not details_df.empty:
                fig = utils.get_spider_plot_fig(details_df)
        except Exception as e:
            print(f"Error creating spider plot: {e}")


        # details_df.to_pickle(ROOT / "pkl" / "details_df_test.pkl")
        print(f"DEBUG ➡ selected ticker={ticker}, name={name}")
        return (
            gr.update(), # last_result_df,
            gr.update(), # pagination_label,
            gr.update(), # page_state,
            #summary_display,
            gr.update(),
            gr.update(selected=1),
            gr.update(value=f"## {name}"),
            gr.update(value=summary),
            gr.update(value=details_df),
            gr.update(value=fig)
        )


    output_df.select(
        on_table_select,
        inputs=[],
        outputs=[
            output_df, pagination_label, page_state, summary_display,
            main_tabs, company_title, company_summary, company_details, company_plot
        ]
    )

    # — Update company‑details whenever the table’s first row changes —
    def on_df_first_row_change(df: pd.DataFrame):
        global selected_ticker
        # if table is empty, do nothing
        if df is None or df.empty:
            return gr.update(), gr.update(), gr.update()
        # extract ticker from first row
        ticker_col = rename_columns.get('ticker','ticker')
        new_ticker = df.iloc[0][ticker_col]
        # if it really changed, fetch new info
        if new_ticker != selected_ticker:
            selected_ticker = new_ticker
            maestro_details = maestro[company_details_cols].copy()
            name, summary, details_df = utils.get_company_info(maestro_details, selected_ticker, rename_columns)
            
            # Create spider plot figure
            fig = None
            try:
                if not details_df.empty:
                    fig = utils.get_spider_plot_fig(details_df)
            except Exception as e:
                print(f"Error creating spider plot: {e}")
            
            return (
                gr.update(value=f"## {name}"),
                gr.update(value=summary),
                gr.update(value=details_df),
                gr.update(value=fig),
                # gr.update()
            )
            
        # otherwise leave components as‑is
        return gr.update(), gr.update(), gr.update(), gr.update()
    output_df.change(
        on_df_first_row_change,
        inputs=[output_df],
        outputs=[company_title, company_summary, company_details, company_plot]
    )

    # ---------------------- EXCLUSION FILTER TOGGLES --------------------------------
    # De momento excluimos esta funcionalidad, al menos en la tabla de acciones, 
    # por la complejidad que añade (es una herencia del buscador de fondos de inversión)
    # Potencial mejora para cuando incorporemos la tabla de ETFs
    '''
    with gr.Row():
        toggle_components = [
            gr.Checkbox(value=True, label=rename_columns.get(k, k)) for k in filtros_keys
        ]
    '''

    # ---------------------- HELPERS ---------------------------------------
    def reset_page():
        return 1

    def prev_page(p):
        return max(p - 1, 1)

    def next_page(p):
        return p + 1

    def search_inputs():
        return [theme_input, ticker_input, page_state]

    def random_action() -> tuple[str,int,str]:
        return utils.random_ticker(maestro), 1, ""

    # ---------------------- BINDINGS --------------------------------------
    # search_dynamic -> search_all
    inputs = [theme_input, ticker_input, page_state]

    buscar_button.click(
        search_all,
        inputs=inputs,
        outputs=[output_df, pagination_label, ticker_input, theme_input, summary_display]
    ).then(
        on_company_tab,
        inputs=[],
        outputs=[company_title, company_summary, company_details, company_plot]
    )    

    ticker_input.submit(
        reset_page, [], page_state
    ).then(
        search_all,
        inputs=inputs,
        outputs=[output_df, pagination_label, ticker_input, theme_input, summary_display]
    ).then(
        on_company_tab,
        inputs=[],
        outputs=[company_title, company_summary, company_details, company_plot]
    )

    theme_input.submit(
        reset_page, [], page_state
    ).then(
        search_all,
        inputs=inputs,
        outputs=[output_df, pagination_label, ticker_input, theme_input, summary_display]
    )

    random_button.click(
        random_action,
        [],
        [ticker_input, page_state, theme_input]
    ).then(
        search_all,
        inputs=inputs,
        outputs=[output_df, pagination_label, ticker_input, theme_input, summary_display]
    )

    reset_button.click(
        reset_initial,
        [],
        [output_df, pagination_label, page_state, ticker_input, theme_input, sort_col, summary_display]
    )

    btn_prev.click(
        prev_page, page_state, page_state
    ).then(
        page_change,
        inputs=inputs,
        outputs=[output_df, pagination_label, ticker_input, theme_input, summary_display]
    )

    btn_next.click(
        next_page, page_state, page_state
    ).then(
        page_change,
        inputs=inputs,
        outputs=[output_df, pagination_label, ticker_input, theme_input, summary_display]
    )

    sort_col.change(
        apply_sort,
        inputs=[sort_col, sort_dir],
        outputs=[output_df, pagination_label, page_state, summary_display]
    )

    sort_dir.change(
        apply_sort,
        inputs=[sort_col, sort_dir],
        outputs=[output_df, pagination_label, page_state, summary_display]
    )


    def on_tab_change(tab_index):
        if tab_index == 1 and selected_ticker:
            maestro_details = maestro[company_details_cols].copy()
            name, summary, details_df = utils.get_company_info(maestro_details, selected_ticker, rename_columns)
            
            # Create spider plot figure
            fig = None
            try:
                if not details_df.empty:
                    fig = utils.get_spider_plot_fig(details_df)
            except Exception as e:
                print(f"Error creating spider plot: {e}")
            
            return (
                gr.update(value=f"## {name}"),
                gr.update(value=summary),
                gr.update(value=details_df),
                gr.update(value=fig)
            )
        return gr.update(), gr.update(), gr.update(), gr.update()


    # ---------------------- FILTERS BY COLUMN ------------------ #
    filterable_columns = [rename_columns.get(c, c) for c in cat_cols]


    def filter_by_column(evt: gr.SelectData) -> tuple[pd.DataFrame,str,int,str]:
        global last_result_df, last_column_filters
        if last_result_df.empty:
            return pd.DataFrame(), "Page 1 of 1", 1, _compose_summary()

        col = last_result_df.columns[evt.index[1]]
        # print(f"DEBUG: resolving to column #{evt.index[1]} → '{col}'")
        val = evt.value
        last_column_filters.append((col, val))
        filtered = last_result_df[last_result_df[col] == val]
        last_result_df = filtered.copy()
        slice_df, label = _paginate(last_result_df, 1)
        summary = _compose_summary()
        return slice_df, label, 1, summary

# ------------------------------------------------------------------------------
# LAUNCH ----------------------------------------------------------------------
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    front.launch()