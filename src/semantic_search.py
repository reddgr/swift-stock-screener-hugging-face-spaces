import duckdb
from sentence_transformers import SentenceTransformer
import pandas as pd
import re

def duckdb_vss_local(
    model: SentenceTransformer,
    duckdb_connection: duckdb.DuckDBPyConnection,
    query: str,
    k: int = 1000,
    brevity_penalty: float = 0.0,
    min_length: int = 131,
    reward_for_literal: float = 0.0,
    first_term_reward: float = 20.0,
    partial_match_factor: float = 0.5,
    table_name: str = "maestro_vector_table",
    embedding_column: str = "vec",
):

    query_vector = model.encode(query)
    embedding_dim = model.get_sentence_embedding_dimension()

    sql = f"""
        SELECT 
            *,
            array_cosine_distance(
                {embedding_column}::float[{embedding_dim}], 
                {query_vector.tolist()}::float[{embedding_dim}]
            ) as distance
        FROM {table_name}
        ORDER BY distance
        LIMIT {k}
    """
    result = duckdb_connection.sql(sql).to_df()
    # Utilizar los parámetros "debug" para mostrar columnas intermedias:
    if brevity_penalty > 0:
        result = penalize_short_summaries(result, factor = brevity_penalty, distance_column = 'distance', 
                                          summary_column = 'longBusinessSummary', min_length=min_length, debug = False)
    if reward_for_literal > 0:
        result = reward_literals(result, query, factor = reward_for_literal, 
                                 partial_match_factor= partial_match_factor, first_term_reward=first_term_reward, distance_column = 'distance', 
                                 summary_column = 'longBusinessSummary', debug = False)

    return result

def penalize_short_summaries(
    df: pd.DataFrame,
    factor: float = 0.1,
    distance_column: str = 'distance',
    summary_column: str = 'longBusinessSummary',
    debug: bool = True,
    min_length: int = 131
    ) -> pd.DataFrame:

    result_df = df.copy()
    result_df['summary_length'] = result_df[summary_column].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    avg_length = max(1.0, result_df['summary_length'].mean())
    max_dist = result_df['distance'].max()

    result_df['percent_shorter'] = result_df['summary_length'].apply(
        lambda x: max(0, (avg_length - x) / avg_length)
    )

    result_df['orig_distance'] = result_df[distance_column]

    # Asignar distancia máxima para resúmenes más cortos que min_length
    # y aplicar penalización proporcional para el resto
    result_df[distance_column] = result_df.apply(
        lambda row: max_dist if row['summary_length'] < min_length else 
                   min(max_dist, row[distance_column] + (row['percent_shorter'] * factor)),
        axis=1
    )
    
    if not debug:
        result_df = result_df.drop(['orig_distance', 'summary_length', 'percent_shorter'], axis=1)

    result_df = result_df.sort_values(by=distance_column, ascending=True) 
    return result_df

def reward_literals(
    df: pd.DataFrame,
    query: str,
    factor: float = 0.1,
    partial_match_factor: float = 0.5,
    first_term_reward: float = 20.0,
    distance_column: str = 'distance',
    summary_column: str = 'longBusinessSummary',
    debug: bool = True
    ) -> pd.DataFrame:

    result_df = df.copy()
    query_lower = query.lower().strip()
    
    def count_phrase_occurrences(summary):
        if pd.isna(summary):
            return 0
        summary_lower = str(summary).lower()
        # Extraemos la primera palabra del resumen y la limpiamos de caracteres especiales 
        # Por ejemplo: "Grifols, S.A. operates as a plasma therapeutic company..." -> Extrae "Grifols", no "Grifols,"
        first_word = summary_lower.split()[0] if summary_lower.strip() and len(summary_lower.split()) > 0 else ""
        first_term = re.sub(r'[^\w\s]', '', first_word.lower())
        # Comprobamos si la primera palabra coincide con la consulta (típicamente el nombre de la empresa)
        _first_term_reward = first_term_reward if first_term == query_lower else 0
        
        # Cuenta coincidencias exactas (palabras completas)
        exact_pattern = r'\b' + re.escape(query_lower) + r'\b'
        exact_count = len(re.findall(exact_pattern, summary_lower))
        
        # Cuenta coincidencias parciales basadas en el tipo de consulta
        if ' ' in query_lower:  # Si la consulta incluye varias palabras
            # Para frases, contamos las veces que aparece en el texto
            partial_pattern = re.escape(query_lower)
            partial_count = len(re.findall(partial_pattern, summary_lower))
        else:
            # Para consultas de una sola palabra, buscamos subcadenas dentro de palabras
            partial_pattern = r'\b\w*' + re.escape(query_lower) + r'\w*\b'
            partial_count = len(re.findall(partial_pattern, summary_lower))
        
        # Resta las coincidencias exactas de las parciales para evitar contar dos veces
        partial_count = partial_count - exact_count
        
        # Penalizamos las coincidencias parciales:
        return _first_term_reward + exact_count + (partial_count * partial_match_factor)
        
    result_df['term_occurrences'] = result_df[summary_column].apply(count_phrase_occurrences)
    result_df['orig_distance'] = result_df[distance_column]
    result_df[distance_column] = result_df.apply(
        lambda row: max(0, row[distance_column] - (row['term_occurrences'] * factor)),
        axis=1
    )
    if not debug:
        result_df = result_df.drop(['orig_distance', 'term_occurrences'], axis=1)
    result_df = result_df.sort_values(by=distance_column, ascending=True)

    return result_df

