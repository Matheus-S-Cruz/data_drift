# Trabalho 10 — Data Drift

Análise de **Data Drift** (desvio de dados) em um cenário climático, comparando
um conjunto de dados de referência (histórico) com um conjunto de dados de
produção. O projeto gera os dados, aplica testes estatísticos para detectar
mudanças de distribuição e produz relatórios automáticos (JSON, executivo e
visual).

## Objetivo da atividade

Identificar automaticamente quando as distribuições das variáveis em produção
divergem das distribuições de referência, indicando se é recomendável
investigar ou **retreinar** um modelo de Machine Learning.

## O que é Data Drift?

**Data Drift** ocorre quando as propriedades estatísticas dos dados de entrada
mudam ao longo do tempo em relação aos dados usados para treinar/validar um
modelo. Como os modelos aprendem padrões a partir dos dados históricos, essas
mudanças podem degradar o desempenho em produção sem que haja qualquer erro de
código. Monitorar o drift é uma prática essencial de **MLOps**.

## Estrutura das pastas

```text
data_drift/
├── data/
│   ├── reference_data.csv      # Dados de referência (1.000 registros)
│   ├── current_data.csv        # Dados de produção (1.000 registros)
│   └── drift_summary.json      # Resultados da detecção (gerado)
├── src/
│   ├── generate_data.py        # Geração dos datasets
│   ├── detect_drift.py         # Testes estatísticos (KS-Test e Qui-quadrado)
│   ├── report_generator.py     # Geração dos relatórios JSON e HTML
│   └── main.py                 # Pipeline completo (orquestração)
├── reports/
│   └── exec_summary.html       # Relatório executivo (gerado)
├── drift_report.html           # Dashboard visual interativo (gerado)
├── requirements.txt
├── README.md
└── .gitignore
```

## Tecnologias e bibliotecas

- **Python 3.11+**
- **pandas** — manipulação dos dados.
- **numpy** — geração dos dados aleatórios.
- **scipy** — testes estatísticos (`ks_2samp`, `chi2_contingency`).
- **plotly** — dashboard visual interativo.

## Instalação

```bash
# Criar e ativar o ambiente virtual
python -m venv .venv

# Linux/Mac
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate

# Instalar dependências
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Como executar

### Opção 1 — Pipeline completo (recomendado)

Executa todas as etapas em sequência (geração dos dados → detecção →
JSON → relatórios HTML):

```bash
python src/main.py
```

### Opção 2 — Etapas individuais

```bash
python src/generate_data.py   # 1. Gera os CSVs em data/
python src/detect_drift.py    # 2. Executa a detecção e imprime os resultados
```

> O `detect_drift.py` isolado apenas imprime os resultados no terminal.
> Para gerar os arquivos de relatório, use `src/main.py`.

## Visualizar os relatórios

- `reports/exec_summary.html` — relatório executivo (tabelas e resumo).
- `drift_report.html` — dashboard visual interativo com histogramas e barras.

Basta abrir os arquivos `.html` em qualquer navegador. (No VS Code, a extensão
*Live Server* também funciona.)

## Testes estatísticos utilizados

| Tipo de variável | Variáveis | Teste | Por quê |
|---|---|---|---|
| Numérica | `temperature`, `humidity`, `wind_speed`, `precipitation` | **Kolmogorov-Smirnov (KS-Test de 2 amostras)** | Compara duas distribuições contínuas sem supor normalidade. |
| Categórica | `sky_condition` | **Qui-quadrado (Chi-square)** | Compara a distribuição de frequências entre categorias. O KS-Test **não** é adequado para dados categóricos. |

- **Nível de significância:** `alpha = 0.05`.
- **Valores ausentes (NaN):** são removidos antes da aplicação dos testes e
  reportados separadamente (quantidade e percentual).
- **Regra de decisão:** se `p-valor < 0.05`, rejeita-se a hipótese de que as
  duas amostras vêm da mesma distribuição → **drift detectado**.

## Como interpretar os resultados

- **P-valor baixo (< 0.05):** há evidência de que a distribuição mudou → drift.
- **P-valor alto (≥ 0.05):** não há evidência de mudança → variável estável.
- **Estatística do teste:** quanto maior, maior a diferença entre as
  distribuições (KS varia de 0 a 1; Qui-quadrado é não negativo).
- **Percentual de variáveis com drift:** indica a severidade geral do desvio.

## Relatórios gerados

1. **JSON (`data/drift_summary.json`)** — resultados completos e reproduzíveis:
   data/hora da execução, nº de registros, teste utilizado, estatística,
   p-valor e status por variável, nº e % de variáveis com drift, informações de
   valores ausentes e conclusão geral. Todos os valores vêm diretamente dos
   testes (nada é preenchido manualmente).
2. **Executivo (`reports/exec_summary.html`)** — objetivo, resumo dos dados,
   tabela por variável, distribuição da categórica, análise de ausentes,
   conclusão e recomendações.
3. **Visual (`drift_report.html`)** — dashboard Plotly interativo com
   histogramas das variáveis numéricas e barras da variável categórica,
   sobrepondo referência x produção.

## Recomendações quando houver drift

- Investigar as variáveis afetadas para identificar a causa raiz.
- Avaliar o impacto no desempenho do modelo em produção.
- Retreinar o modelo com dados recentes quando o drift for significativo.
- Verificar a origem de valores ausentes no pipeline de dados.
- Manter o monitoramento contínuo das distribuições.
