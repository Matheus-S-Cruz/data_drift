"""Detecção de Data Drift entre os conjuntos de referência e produção.

Variáveis numéricas  -> Teste de Kolmogorov-Smirnov (KS-Test de 2 amostras).
Variável categórica  -> Teste Qui-quadrado (chi-square) de homogeneidade.

Todos os testes usam nível de significância alpha = 0.05. Os valores ausentes
são tratados (removidos) antes da aplicação dos testes numéricos e reportados
separadamente.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

# Nível de significância
ALPHA = 0.05

# Classificação das variáveis
NUMERIC_VARS = ["temperature", "humidity", "wind_speed", "precipitation"]
CATEGORICAL_VARS = ["sky_condition"]


def _missing_info(series: pd.Series) -> dict:
    """Retorna a quantidade e o percentual de valores ausentes de uma série."""
    total = len(series)
    missing = int(series.isna().sum())
    pct = (missing / total * 100) if total else 0.0
    return {"count": missing, "percent": round(pct, 2)}


def analyze_numeric(ref: pd.Series, curr: pd.Series, name: str) -> dict:
    """Aplica o KS-Test em uma variável numérica, tratando valores ausentes."""
    ref_missing = _missing_info(ref)
    curr_missing = _missing_info(curr)

    # Remove NaN antes do teste (KS-Test não aceita valores ausentes)
    ref_clean = ref.dropna()
    curr_clean = curr.dropna()

    stat, p_value = ks_2samp(ref_clean, curr_clean)
    has_drift = bool(p_value < ALPHA)

    return {
        "variable": name,
        "type": "numeric",
        "test": "Kolmogorov-Smirnov (KS-Test)",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "has_drift": has_drift,
        "status": "DRIFT DETECTADO" if has_drift else "Estável",
        "missing": {"reference": ref_missing, "current": curr_missing},
    }


def analyze_categorical(ref: pd.Series, curr: pd.Series, name: str) -> dict:
    """Aplica o teste Qui-quadrado de homogeneidade em uma variável categórica."""
    ref_missing = _missing_info(ref)
    curr_missing = _missing_info(curr)

    ref_clean = ref.dropna()
    curr_clean = curr.dropna()

    # Tabela de contingência: categorias x (referência / produção)
    categories = sorted(set(ref_clean.unique()) | set(curr_clean.unique()))
    ref_counts = ref_clean.value_counts().reindex(categories, fill_value=0)
    curr_counts = curr_clean.value_counts().reindex(categories, fill_value=0)
    contingency = pd.DataFrame({"reference": ref_counts, "current": curr_counts})

    stat, p_value, dof, _ = chi2_contingency(contingency.values)
    has_drift = bool(p_value < ALPHA)

    # Distribuição percentual por categoria (para os relatórios)
    ref_dist = (ref_counts / ref_counts.sum() * 100).round(2).to_dict()
    curr_dist = (curr_counts / curr_counts.sum() * 100).round(2).to_dict()

    return {
        "variable": name,
        "type": "categorical",
        "test": "Qui-quadrado (Chi-square)",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": int(dof),
        "has_drift": has_drift,
        "status": "DRIFT DETECTADO" if has_drift else "Estável",
        "missing": {"reference": ref_missing, "current": curr_missing},
        "distribution": {"reference": ref_dist, "current": curr_dist},
    }


def run_drift_analysis(ref_df: pd.DataFrame | None = None,
                       curr_df: pd.DataFrame | None = None) -> dict:
    """Executa a análise completa de drift e retorna um dicionário de resultados.

    Se os DataFrames não forem informados, eles são carregados de `data/`.
    """
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"

    if ref_df is None or curr_df is None:
        print("Carregando datasets...")
        ref_df = pd.read_csv(data_dir / "reference_data.csv")
        curr_df = pd.read_csv(data_dir / "current_data.csv")

    results = []

    print("\n--- VARIÁVEIS NUMÉRICAS (KS-Test, alpha = 0.05) ---")
    for col in NUMERIC_VARS:
        res = analyze_numeric(ref_df[col], curr_df[col], col)
        results.append(res)
        print(f"{col:15} | stat={res['statistic']:.4f} | p={res['p_value']:.5f} "
              f"| {res['status']}")

    print("\n--- VARIÁVEL CATEGÓRICA (Qui-quadrado, alpha = 0.05) ---")
    for col in CATEGORICAL_VARS:
        res = analyze_categorical(ref_df[col], curr_df[col], col)
        results.append(res)
        print(f"{col:15} | stat={res['statistic']:.4f} | p={res['p_value']:.5f} "
              f"| {res['status']}")

    # Métricas agregadas
    total_vars = len(results)
    drifted_vars = sum(1 for r in results if r["has_drift"])
    drift_pct = round(drifted_vars / total_vars * 100, 2) if total_vars else 0.0

    # Conclusão / recomendação
    if drift_pct == 0:
        conclusion = ("Nenhuma variável apresentou drift. Não é necessário "
                      "investigar ou retreinar o modelo no momento.")
    elif drift_pct < 50:
        conclusion = (f"{drifted_vars} de {total_vars} variáveis apresentaram drift "
                      f"({drift_pct}%). Recomenda-se investigar as variáveis afetadas "
                      f"e monitorar o desempenho do modelo.")
    else:
        conclusion = (f"{drifted_vars} de {total_vars} variáveis apresentaram drift "
                      f"({drift_pct}%). É fortemente recomendado investigar as causas "
                      f"e retreinar o modelo com dados atualizados.")

    summary = {
        "n_reference_records": int(len(ref_df)),
        "n_current_records": int(len(curr_df)),
        "alpha": ALPHA,
        "results": results,
        "n_variables_analyzed": total_vars,
        "n_variables_with_drift": drifted_vars,
        "drift_percentage": drift_pct,
        "conclusion": conclusion,
    }

    print(f"\nVariáveis com drift: {drifted_vars}/{total_vars} ({drift_pct}%)")
    print(f"Conclusão: {conclusion}")

    return summary


if __name__ == "__main__":
    run_drift_analysis()
