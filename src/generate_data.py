"""Gera os conjuntos de dados de referência e produção para a análise de Data Drift.

Cada conjunto possui 1.000 registros com quatro variáveis numéricas
(temperature, humidity, wind_speed, precipitation) e uma variável categórica
(sky_condition). Os dados de produção contêm diferenças de distribuição
intencionais (drift) e 5% de valores ausentes na variável `temperature`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Constantes de configuração
N_SAMPLES = 1000
MISSING_RATE = 0.05  # 5% de valores ausentes em temperature (produção)
RANDOM_SEED = 42

# Categorias possíveis para sky_condition
SKY_CATEGORIES = ["clear", "cloudy", "rainy"]
# Proporções (clear, cloudy, rainy)
REF_SKY_PROBS = [0.60, 0.30, 0.10]   # Referência: predominância de céu limpo
CURR_SKY_PROBS = [0.10, 0.30, 0.60]  # Produção: predominância de chuva (drift)


def _build_reference(rng: np.random.Generator) -> pd.DataFrame:
    """Cria o conjunto de referência (clima normal)."""
    return pd.DataFrame(
        {
            "temperature": rng.normal(loc=25.0, scale=3.0, size=N_SAMPLES),
            "humidity": rng.normal(loc=70.0, scale=10.0, size=N_SAMPLES),
            "wind_speed": rng.normal(loc=15.0, scale=4.0, size=N_SAMPLES),
            "precipitation": rng.exponential(scale=2.0, size=N_SAMPLES),
            "sky_condition": rng.choice(SKY_CATEGORIES, size=N_SAMPLES, p=REF_SKY_PROBS),
        }
    )


def _build_current(rng: np.random.Generator) -> pd.DataFrame:
    """Cria o conjunto de produção com data drift (onda de calor, seca, mais chuva)."""
    curr = pd.DataFrame(
        {
            "temperature": rng.normal(loc=34.0, scale=4.0, size=N_SAMPLES),   # Drift (+9°C)
            "humidity": rng.normal(loc=45.0, scale=12.0, size=N_SAMPLES),     # Drift (-25%)
            "wind_speed": rng.normal(loc=15.5, scale=4.1, size=N_SAMPLES),    # Estável
            "precipitation": rng.exponential(scale=0.5, size=N_SAMPLES),      # Drift (queda)
            "sky_condition": rng.choice(SKY_CATEGORIES, size=N_SAMPLES, p=CURR_SKY_PROBS),
        }
    )

    # Insere 5% de valores ausentes (NaN) em temperature na produção
    n_missing = int(round(N_SAMPLES * MISSING_RATE))
    missing_idx = rng.choice(N_SAMPLES, size=n_missing, replace=False)
    curr.loc[missing_idx, "temperature"] = np.nan

    return curr


def generate_weather_datasets() -> None:
    """Gera e salva os datasets de referência e produção em `data/`."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    ref_data = _build_reference(rng)
    curr_data = _build_current(rng)

    ref_path = data_dir / "reference_data.csv"
    curr_path = data_dir / "current_data.csv"
    ref_data.to_csv(ref_path, index=False)
    curr_data.to_csv(curr_path, index=False)

    n_missing = int(curr_data["temperature"].isna().sum())
    print("Datasets climáticos gerados em 'data/':")
    print(f"  - {ref_path.name}: {len(ref_data)} registros")
    print(f"  - {curr_path.name}: {len(curr_data)} registros")
    print(f"  - Valores ausentes em 'temperature' (produção): {n_missing} "
          f"({n_missing / N_SAMPLES:.1%})")


if __name__ == "__main__":
    generate_weather_datasets()
