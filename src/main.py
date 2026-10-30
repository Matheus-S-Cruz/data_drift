"""Script principal: executa todo o pipeline de análise de Data Drift.

Sequência:
  1. Gera os dados de referência e produção.
  2. Executa a detecção de drift (KS-Test + Qui-quadrado).
  3. Gera o relatório JSON (data/drift_summary.json).
  4. Gera os relatórios HTML (executivo e visual).
"""

from pathlib import Path

import pandas as pd

from detect_drift import run_drift_analysis
from generate_data import generate_weather_datasets
from report_generator import (
    generate_exec_summary,
    generate_json_report,
    generate_visual_report,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    reports_dir = base_dir / "reports"

    print("=" * 60)
    print("PIPELINE DE ANÁLISE DE DATA DRIFT")
    print("=" * 60)

    # 1. Geração dos dados
    print("\n[1/4] Gerando datasets...")
    generate_weather_datasets()

    # Carrega os dados uma única vez para reutilizar nas etapas seguintes
    ref_df = pd.read_csv(data_dir / "reference_data.csv")
    curr_df = pd.read_csv(data_dir / "current_data.csv")

    # 2. Detecção de drift
    print("\n[2/4] Executando detecção de drift...")
    summary = run_drift_analysis(ref_df, curr_df)

    # 3. Relatório JSON
    print("\n[3/4] Gerando relatório JSON...")
    generate_json_report(summary, data_dir / "drift_summary.json")

    # 4. Relatórios HTML
    print("\n[4/4] Gerando relatórios HTML...")
    generate_exec_summary(summary, reports_dir / "exec_summary.html")
    generate_visual_report(ref_df, curr_df, summary, base_dir / "drift_report.html")

    print("\n" + "=" * 60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    main()
