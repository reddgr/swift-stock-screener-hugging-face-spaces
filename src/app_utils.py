import pandas as pd
from typing import Tuple
import numpy as np
import plotly.graph_objects as go
import re

_NEG_COLOR = "red"

def format_large_number(n, decimals=2):
    if n >= 1e12:
        return f'{n / 1e12:.{decimals}f} T'
    elif n >= 1e9:
        return f'{n / 1e9:.{decimals}f} B'
    elif n >= 1e6:
        return f'{n / 1e6:.{decimals}f} M'
    else:
        return str(n)

def format_results(df: pd.DataFrame, rename_columns: dict) -> pd.DataFrame:
    # Índice 100
    if "ind_sust" in df.columns:
        df["ind_sust"] = df["ind_sust"].apply(lambda x: "-" if pd.isna(x) else int(round(x * 100, 0)))
    # 1 decimal
    for col in ["trailingPE", "beta"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "-" if pd.isna(x) else f"{x:.1f}")

    # 2 decimales
    if "Search dist." in df.columns:
        df["Search dist."] = df["Search dist."].apply(lambda n: "-" if pd.isna(n) else f"{n:.2f}")

    # Cantidades monetarias grandes
    for col in ["marketCap", "totalRevenue", "ebitda"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda n: "-" if pd.isna(n) else format_large_number(n, 1))
    # Porcentajes 1 decimal
    for col in ["ret_365", "revenueGrowth"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "-" if pd.isna(x) or x == 0 else f"{(x * 100):.1f}%")
    # Porcentajes 1 decimal (porcentaje numérico en fuente)
    for col in ["dividendYield"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "-" if pd.isna(x) else f"{round(x, 1)}%")
    # Volatilidad
    if "vol_365" in df.columns:
        df["vol_365"] = df["vol_365"].apply(lambda x: "-" if pd.isna(x) or x == 0 else f"{x:.4f}")

    # Devolvemos el dataframe con los nombres de columnas renombrados
    return df.rename(columns=rename_columns)


def random_ticker(df: pd.DataFrame) -> str:
    return df["ticker"].sample(n=1).values[0]

def styler_negative_red(df: pd.DataFrame, cols: list[str] | None = None):
    """
    Returns a Styler that paints negative numeric values in *cols*.
    Columns absent in *df* are ignored.
    """
    cols = [c for c in (cols or df.columns) if c in df.columns]

    def _style(v):
        try:
            num = float(re.sub(r"[ %,TMB]", "", str(v)))
            if num < 0:
                return f"color:{_NEG_COLOR}"
        except ValueError:
            pass
        return ""

    return df.style.applymap(_style, subset=cols)

def get_company_info(
    maestro: pd.DataFrame,
    ticker: str,
    rename_columns: dict
) -> Tuple[str, str, pd.DataFrame]:
    """
    Returns the company name, longBusinessSummary, and a DataFrame
    of all other fields for the given ticker.
    """
    company = maestro[maestro["ticker"] == ticker]
    if company.empty:
        return ticker, "No data available.", pd.DataFrame()

    # extract name & summary
    name = company["security"].iloc[0] if "security" in company.columns else ticker
    summary = company["longBusinessSummary"].iloc[0] if "longBusinessSummary" in company.columns else ""

    # build details table
    details = company.drop(columns=["longBusinessSummary"], errors="ignore").iloc[0]
    df = pd.DataFrame({
        "Field": details.index.tolist(),
        "Value": details.values.tolist()
    })
    df["Field"] = df["Field"].map(lambda c: rename_columns.get(c, c))

    # Round _norm fields to 3 decimal places
    for i, field in enumerate(df["Field"]):
        if field.endswith("norm."):
            value = df.iloc[i]["Value"]
            if isinstance(value, (int, float)) and not pd.isna(value):
                df.iloc[i, df.columns.get_loc("Value")] = round(value, 3)
        # Process numeric fields using format_results function
        # Extract numeric fields (excluding already processed _norm fields)
        numeric_fields = []
        numeric_values = []
        numeric_indices = []
        
        for i, (display_field, value) in enumerate(zip(df["Field"], df["Value"])):
            if not display_field.endswith("norm.") and isinstance(value, (int, float)) and not pd.isna(value):
                # Get original field name using inverse rename dictionary
                orig_field = next((k for k, v in rename_columns.items() if v == display_field), display_field)
                numeric_fields.append(orig_field)
                numeric_values.append(value)
                numeric_indices.append(i)
        
        if numeric_fields:
            # Create a single-row dataframe with original field names
            temp_df = pd.DataFrame([numeric_values], columns=numeric_fields)
            
            # Apply format_results function
            formatted_df = format_results(temp_df, rename_columns)
            
            # Put formatted values back into the original dataframe
            for i, field in zip(numeric_indices, numeric_fields):
                display_field = rename_columns.get(field, field)
                df.iloc[i, df.columns.get_loc("Value")] = formatted_df.iloc[0][display_field]
        

    return name, summary, df


def spider_plot(df: pd.DataFrame) -> None:
    spider_plot_cols = ['Beta norm.', 'Debt to Equity norm.', '1-year Return norm.', 'Revenue Growth norm.', 'Volatility norm.']
    plot_data = df[df['Field'].isin(spider_plot_cols)].set_index('Field')
    values = plot_data.loc[spider_plot_cols, 'Value'].fillna(0.5).astype(float).tolist()
    metrics_to_invert = ['Debt to Equity norm.', 'Beta norm.', 'Volatility norm.']
    values = [1 - v if col in metrics_to_invert else v for v, col in zip(values, spider_plot_cols)]
    categories = [s.replace(' norm.', '').replace('1-year', '1yr').replace('Debt to Equity', 'D/E') for s in spider_plot_cols]
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Company Profile'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[0.5] * len(categories) + [0.5], # Append the first r value to close the loop
        theta=categories + [categories[0]], # Append the first theta value to close the loop
        mode='lines',
        line=dict(dash='dot', color='grey'),
        fill='toself', # Keep fill='none' if you only want the line
        fillcolor='rgba(0,0,0,0)', # Make fill transparent if only line is desired
        name='Median (0.5)'
    ))

    legend_text = (
        "<b>Quantile Scale: 0 to 1</b><br>"
        "D/E, Beta, and Volatility:<br>"
        "0 is highest, 1 is lowest<br>"
        "Rev. growth and 1yr return:<br>"
        "0 is lowest, 1 is highest<br>"
    )

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]  # Set the range from 0 to 1
        )),
    showlegend=True,
    title='Normalized Company Metrics',
    annotations=[
            go.layout.Annotation(
                text=legend_text,
                align='right',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.41, 
                y=-0.1 
            )
        ],
        margin=dict(b=120),
        width=600,
        height=500
    )

    fig.show()


# Create a new function in app_utils.py that returns the figure instead of showing it
def get_spider_plot_fig_v0(df: pd.DataFrame):
    spider_plot_cols = ['Beta norm.', 'Debt to Equity norm.', '1-year Return norm.', 'Revenue Growth norm.', 'Volatility norm.']
    plot_data = df[df['Field'].isin(spider_plot_cols)].set_index('Field')
    values = plot_data.loc[spider_plot_cols, 'Value'].fillna(0.5).astype(float).tolist()
    metrics_to_invert = ['Debt to Equity norm.', 'Beta norm.', 'Volatility norm.']
    values = [1 - v if col in metrics_to_invert else v for v, col in zip(values, spider_plot_cols)]
    categories = [s.replace(' norm.', '').replace('1-year', '1yr').replace('Debt to Equity', 'D/E') for s in spider_plot_cols]
    company_name = df.loc[df['Field'] == 'Name', 'Value'].values[0]
    fig = go.Figure()


    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Company Profile'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[0.5] * len(categories) + [0.5], # Append the first r value to close the loop
        theta=categories + [categories[0]], # Append the first theta value to close the loop
        mode='lines',
        line=dict(dash='dot', color='grey'),
        fill='toself', # Keep fill='none' if you only want the line
        fillcolor='rgba(0,0,0,0)', # Make fill transparent if only line is desired
        name='Median (0.5)'
    ))

    legend_text = (
        "<b>Quantile Scale: 0 to 1</b><br>"
        "D/E, Beta, and Volatility:<br>"
        "0 is highest, 1 is lowest<br>"
        "Rev. growth and 1yr return:<br>"
        "0 is lowest, 1 is highest<br>"
    )

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]  # Set the range from 0 to 1
        )),
    showlegend=True,
    title=f'{company_name} - Normalized Metrics',
    annotations=[
            go.layout.Annotation(
                text=legend_text,
                align='right',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.41, 
                y=-0.1 
            )
        ],
        margin=dict(b=120),
        width=600,
        height=500
    )

    return fig  


def get_spider_plot_fig(df: pd.DataFrame):
    spider_plot_cols = ['Beta norm.', 'Debt to Equity norm.', '1-year Return norm.', 'Revenue Growth norm.', 'Volatility norm.']
    plot_data = df[df['Field'].isin(spider_plot_cols)].set_index('Field')
    values = plot_data.loc[spider_plot_cols, 'Value'].fillna(0.5).astype(float).tolist()
    metrics_to_invert = ['Debt to Equity norm.', 'Beta norm.', 'Volatility norm.']
    values = [1 - v if col in metrics_to_invert else v for v, col in zip(values, spider_plot_cols)]
    
    # Calculate average strength score
    avg_strength = round(np.mean(values) * 100)
    
    # Determine color based on score
    if avg_strength < 30:
        profile_color = 'red'
    elif avg_strength < 50:
        profile_color = 'gold'
    elif avg_strength < 60:
        profile_color = 'blue'
    else:
        profile_color = 'green'
    
    categories = [s.replace(' norm.', '').replace('1-year', '1yr').replace('Debt to Equity', 'D/E') for s in spider_plot_cols]
    company_name = df.loc[df['Field'] == 'Name', 'Value'].values[0]
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Company Profile',
        line=dict(color=profile_color),
        fillcolor=f'rgba({",".join(["255,0,0,0.2" if profile_color == "red" else "255,215,0,0.2" if profile_color == "gold" else "0,0,255,0.2" if profile_color == "blue" else "0,128,0,0.2"])})'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[0.5] * len(categories) + [0.5],
        theta=categories + [categories[0]],
        mode='lines',
        line=dict(dash='dot', color='grey'),
        fill='toself',
        fillcolor='rgba(0,0,0,0)',
        name='Median (0.5)'
    ))

    # Determine strength level text based on score
    if avg_strength < 30:
        strength_level = "very low"
        text_color = "red"
    elif avg_strength < 50:
        strength_level = "low"
        text_color = "gold"
    elif avg_strength < 60:
        strength_level = "medium"
        text_color = "blue"
    else:
        strength_level = "high"
        text_color = "green"
        
    legend_text = (
        f"<b>Avg. strength: {avg_strength}</b> (<span style='color:{text_color}'>{strength_level}</span>)<br><br>"
        "<b>Quantile Scale: 0 to 1</b><br>"
        "D/E, Beta, and Volatility:<br>"
        "0 is highest, 1 is lowest<br>"
        "Rev. growth and 1yr return:<br>"
        "0 is lowest, 1 is highest<br>"
    )

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True,
    title=f'{company_name} - Normalized Metrics',
    annotations=[
            go.layout.Annotation(
                text=legend_text,
                align='right',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.41, 
                y=-0.1 
            )
        ],
        margin=dict(b=120),
        width=600,
        height=500
    )

    return fig