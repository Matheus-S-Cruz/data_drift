"""Geração dos relatórios de Data Drift.

Produz três saídas a partir dos resultados da detecção:
  1. JSON      -> data/drift_summary.json
  2. Executivo -> reports/exec_summary.html
  3. Visual    -> drift_report.html (histogramas numéricos + barras da categórica)
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detect_drift import CATEGORICAL_VARS, NUMERIC_VARS


# --------------------------------------------------------------------------- #
# 1. Relatório JSON
# --------------------------------------------------------------------------- #
def generate_json_report(summary: dict, output_path: Path) -> Path:
    """Salva o resumo da análise em um arquivo JSON."""
    payload = {"execution_datetime": datetime.now().isoformat(timespec="seconds")}
    payload.update(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print(f"Relatório JSON gerado: '{output_path}'")
    return output_path


# --------------------------------------------------------------------------- #
# 2. Relatório executivo (HTML)
# --------------------------------------------------------------------------- #
def _format_missing(missing: dict) -> str:
    ref = missing["reference"]
    curr = missing["current"]
    return (f"Ref: {ref['count']} ({ref['percent']}%) | "
            f"Prod: {curr['count']} ({curr['percent']}%)")


def generate_exec_summary(summary: dict, output_path: Path) -> Path:
    """Gera um relatório executivo em HTML a partir do resumo da análise."""
    rows = []
    for r in summary["results"]:
        badge = ("drift" if r["has_drift"] else "stable")
        label = "🚨 DRIFT" if r["has_drift"] else "✅ Estável"
        rows.append(f"""
        <tr>
            <td>{r['variable']}</td>
            <td>{r['type']}</td>
            <td>{r['test']}</td>
            <td>{r['statistic']:.4f}</td>
            <td>{r['p_value']:.5f}</td>
            <td>{_format_missing(r['missing'])}</td>
            <td class="{badge}">{label}</td>
        </tr>""")
    table_rows = "".join(rows)

    drifted = [r["variable"] for r in summary["results"] if r["has_drift"]]
    drifted_list = ", ".join(drifted) if drifted else "Nenhuma"

    # Distribuição da variável categórica (se houver)
    cat_block = ""
    for r in summary["results"]:
        if r["type"] == "categorical" and "distribution" in r:
            ref_d = r["distribution"]["reference"]
            curr_d = r["distribution"]["current"]
            cats = sorted(set(ref_d) | set(curr_d))
            cat_rows = "".join(
                f"<tr><td>{c}</td><td>{ref_d.get(c, 0)}%</td>"
                f"<td>{curr_d.get(c, 0)}%</td></tr>"
                for c in cats
            )
            cat_block = f"""
        <h2>Distribuição de <code>{r['variable']}</code></h2>
        <table>
            <thead><tr><th>Categoria</th><th>Referência</th><th>Produção</th></tr></thead>
            <tbody>{cat_rows}</tbody>
        </table>"""

    # Análise de valores ausentes
    missing_rows = "".join(
        f"<tr><td>{r['variable']}</td>"
        f"<td>{r['missing']['reference']['count']} "
        f"({r['missing']['reference']['percent']}%)</td>"
        f"<td>{r['missing']['current']['count']} "
        f"({r['missing']['current']['percent']}%)</td></tr>"
        for r in summary["results"]
    )

    recommendation = summary["conclusion"]

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Executivo - Data Drift</title>
    <style>
        body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 2rem;
               background: #f5f7fa; color: #222; }}
        .container {{ max-width: 960px; margin: 0 auto; background: #fff;
                     padding: 2rem 2.5rem; border-radius: 10px;
                     box-shadow: 0 2px 10px rgba(0,0,0,.08); }}
        h1 {{ color: #1f3a5f; border-bottom: 3px solid #1f77b4; padding-bottom: .5rem; }}
        h2 {{ color: #1f3a5f; margin-top: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
        th, td {{ border: 1px solid #ddd; padding: .6rem .8rem; text-align: left;
                 font-size: .92rem; }}
        th {{ background: #1f77b4; color: #fff; }}
        tr:nth-child(even) {{ background: #f2f6fb; }}
        .drift {{ color: #c0392b; font-weight: bold; }}
        .stable {{ color: #27ae60; font-weight: bold; }}
        .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }}
        .card {{ flex: 1; min-width: 160px; background: #1f77b4; color: #fff;
                padding: 1rem; border-radius: 8px; text-align: center; }}
        .card .value {{ font-size: 1.8rem; font-weight: bold; }}
        .conclusion {{ background: #fff8e1; border-left: 5px solid #f1c40f;
                      padding: 1rem 1.2rem; margin-top: 1.5rem; border-radius: 4px; }}
        code {{ background: #eef; padding: .1rem .3rem; border-radius: 3px; }}
        .meta {{ color: #666; font-size: .85rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Relatório Executivo — Análise de Data Drift</h1>
        <p class="meta">Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>

        <h2>Objetivo</h2>
        <p>Comparar os dados de produção com os dados de referência (histórico) para
        identificar mudanças na distribuição das variáveis (Data Drift) que possam
        degradar o desempenho de um modelo de Machine Learning.</p>

        <h2>Resumo dos Conjuntos de Dados</h2>
        <div class="cards">
            <div class="card"><div class="value">{summary['n_reference_records']}</div>
                <div>Registros de referência</div></div>
            <div class="card"><div class="value">{summary['n_current_records']}</div>
                <div>Registros de produção</div></div>
            <div class="card"><div class="value">{summary['n_variables_analyzed']}</div>
                <div>Variáveis analisadas</div></div>
            <div class="card"><div class="value">{summary['n_variables_with_drift']}</div>
                <div>Variáveis com drift</div></div>
            <div class="card"><div class="value">{summary['drift_percentage']}%</div>
                <div>% com drift</div></div>
        </div>

        <h2>Resultados por Variável</h2>
        <table>
            <thead>
                <tr>
                    <th>Variável</th><th>Tipo</th><th>Teste</th><th>Estatística</th>
                    <th>P-valor</th><th>Valores ausentes</th><th>Resultado</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>

        <h2>Variáveis com Drift Detectado</h2>
        <p><strong>{drifted_list}</strong></p>
        {cat_block}

        <h2>Análise de Valores Ausentes</h2>
        <table>
            <thead><tr><th>Variável</th><th>Referência</th><th>Produção</th></tr></thead>
            <tbody>{missing_rows}</tbody>
        </table>

        <h2>Conclusão e Recomendações</h2>
        <div class="conclusion">
            <p>{recommendation}</p>
            <ul>
                <li>Investigar as variáveis que apresentaram drift para entender a causa raiz.</li>
                <li>Avaliar o impacto das mudanças no desempenho do modelo em produção.</li>
                <li>Considerar o retreinamento do modelo com dados recentes caso o drift seja significativo.</li>
                <li>Verificar a origem dos valores ausentes no pipeline de dados.</li>
                <li>Manter o monitoramento contínuo das distribuições.</li>
            </ul>
        </div>
    </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(html)

    print(f"Relatório executivo gerado: '{output_path}'")
    return output_path


# --------------------------------------------------------------------------- #
# 3. Relatório visual (Plotly) — numéricas + categórica
# --------------------------------------------------------------------------- #
def generate_visual_report(ref_df: pd.DataFrame, curr_df: pd.DataFrame,
                           summary: dict, output_path: Path) -> Path:
    """Gera o dashboard visual interativo com histogramas e barras."""
    status_by_var = {r["variable"]: r for r in summary["results"]}

    all_vars = NUMERIC_VARS + CATEGORICAL_VARS
    n = len(all_vars)
    cols = 2
    rows = (n + cols - 1) // cols

    titles = []
    for v in all_vars:
        r = status_by_var[v]
        tag = "🚨 DRIFT" if r["has_drift"] else "✅ Estável"
        titles.append(f"{v.upper()} ({tag})")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        vertical_spacing=0.12, horizontal_spacing=0.1)

    for idx, var in enumerate(all_vars):
        row = idx // cols + 1
        col = idx % cols + 1
        first = idx == 0

        if var in NUMERIC_VARS:
            fig.add_trace(
                go.Histogram(x=ref_df[var].dropna(), name="Referência",
                             legendgroup="ref", showlegend=first,
                             marker_color="#1f77b4", opacity=0.6),
                row=row, col=col)
            fig.add_trace(
                go.Histogram(x=curr_df[var].dropna(), name="Produção",
                             legendgroup="curr", showlegend=first,
                             marker_color="#ff7f0e", opacity=0.6),
                row=row, col=col)
        else:
            # Variável categórica -> gráfico de barras com percentuais
            dist = status_by_var[var]["distribution"]
            cats = sorted(set(dist["reference"]) | set(dist["current"]))
            fig.add_trace(
                go.Bar(x=cats, y=[dist["reference"].get(c, 0) for c in cats],
                       name="Referência", legendgroup="ref", showlegend=False,
                       marker_color="#1f77b4", opacity=0.8),
                row=row, col=col)
            fig.add_trace(
                go.Bar(x=cats, y=[dist["current"].get(c, 0) for c in cats],
                       name="Produção", legendgroup="curr", showlegend=False,
                       marker_color="#ff7f0e", opacity=0.8),
                row=row, col=col)

    fig.update_layout(
        title_text=(f"Dashboard MLOps: Análise de Data Drift Climático "
                    f"— {summary['n_variables_with_drift']}/"
                    f"{summary['n_variables_analyzed']} variáveis com drift "
                    f"({summary['drift_percentage']}%)"),
        barmode="overlay",
        template="plotly_white",
        height=350 * rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"Relatório visual gerado: '{output_path}'")
    return output_path
