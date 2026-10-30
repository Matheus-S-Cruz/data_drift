from pathlib import Path

import numpy as np
import pandas as pd


def generate_weather_datasets():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 1000

    # 1. Dados de referência (clima normal)
    ref_data = pd.DataFrame(
        {
            "temperature": np.random.normal(loc=25.0, scale=3.0, size=n_samples),
            "humidity": np.random.normal(loc=70.0, scale=10.0, size=n_samples),
            "wind_speed": np.random.normal(loc=15.0, scale=4.0, size=n_samples),
            "precipitation": np.random.exponential(scale=2.0, size=n_samples),
        }
    )

    # 2. Dados atuais / produção (com data drift: onda de calor e seca)
    curr_data = pd.DataFrame(
        {
            "temperature": np.random.normal(loc=34.0, scale=4.0, size=n_samples),  # Drift (+9C)
            "humidity": np.random.normal(loc=45.0, scale=12.0, size=n_samples),  # Drift (-25%)
            "wind_speed": np.random.normal(loc=15.5, scale=4.1, size=n_samples),  # Estável
            "precipitation": np.random.exponential(scale=0.5, size=n_samples),  # Drift (queda)
        }
    )

    ref_data.to_csv(data_dir / "reference_data.csv", index=False)
    curr_data.to_csv(data_dir / "current_data.csv", index=False)
    print("Datasets climáticos criados em 'data/'!")


if __name__ == "__main__":
    generate_weather_datasets()