from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp


def run_drift_analysis():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"

    print("Carregando datasets...")
    ref_df = pd.read_csv(data_dir / "reference_data.csv")
    curr_df = pd.read_csv(data_dir / "current_data.csv")

    print("\n--- ANÁLISE ESTATÍSTICA DIRETA (KS-TEST) ---")
    p_value_threshold = 0.05
    results = []

    for column in ref_df.columns:
        stat, p_value = ks_2samp(ref_df[column], curr_df[column])
        has_drift = p_value < p_value_threshold
        status = "DRIFT DETECTADO 🚨" if has_drift else "Estável ✅"

        results.append(
            {
                "variable": column,
                "stat": stat,
                "p_value": p_value,
                "has_drift": has_drift,
                "status": status,
            }
        )

        print(f"Variável: {column:15} | p-value: {p_value:.5f} | Status: {status}")

    print("\nGerando relatório visual interativo...")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[col.upper() for col in ref_df.columns],
        vertical_spacing=0.12,
    )
    coords = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, col in enumerate(ref_df.columns):
        row, col_idx = coords[idx]

        fig.add_trace(
            go.Histogram(
                x=ref_df[col],
                name=f"{col} (Ref)",
                legendgroup="ref",
                showlegend=(idx == 0),
                marker_color="#1f77b4",
                opacity=0.6,
            ),
            row=row,
            col=col_idx,
        )

        fig.add_trace(
            go.Histogram(
                x=curr_df[col],
                name=f"{col} (Atual)",
                legendgroup="curr",
                showlegend=(idx == 0),
                marker_color="#ff7f0e",
                opacity=0.6,
            ),
            row=row,
            col=col_idx,
        )

    fig.update_layout(
        title_text="Dashboard MLOps: Análise de Data Drift Climático",
        barmode="overlay",
        template="plotly_white",
        height=700,
    )

    report_path = base_dir / "drift_report.html"
    fig.write_html(report_path)
    print(f"Relatório gerado com sucesso: '{report_path.name}'")


if __name__ == "__main__":
    run_drift_analysis()
