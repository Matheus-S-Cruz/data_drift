# Para Rodar
python -m venv .venv
.\.venv\Scripts\Activate

## Instalar Dependências
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

## Executar
python src/generate_data.py
python src/detect_drift.py

## Visualizar
1. Baixe a extensão "Live Server"
2. Abra "drift_report.html" com Live Server