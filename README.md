# Algoritmo Genético — Conversão Rust → Python

Implementação de um framework de Algoritmo Genético (AG) em Python, convertida a partir do projeto original em Rust (`alex/`). O código preserva os algoritmos, a separação de responsabilidades e os parâmetros originais.

---

## Estrutura do projeto

```
trabalho-1/
├── main.py                 # Ponto de entrada
├── requirements.txt        # Dependências Python
├── config/
│   ├── config_ex1.json     # Parâmetros do EX1
│   ├── config_ex2.json     # Parâmetros do EX2
│   └── config_ex3.json     # Parâmetros do EX3
├── data/
│   └── ex3.cnf             # Instância 3-SAT (100 variáveis, 430 cláusulas)
├── output/
│   ├── ex1/                # Gráficos gerados pelo EX1
│   ├── ex2/                # Gráficos gerados pelo EX2
│   └── ex3/                # Gráficos gerados pelo EX3
└── src/
    ├── __init__.py
    ├── ae.py               # Motor do algoritmo genético
    ├── ex1.py              # EX1 — otimização de função algébrica
    ├── ex2.py              # EX2 — problema dos rádios (com restrição)
    ├── ex3.py              # EX3 — 3-SAT
    └── grafico.py          # Geração de gráficos de convergência (matplotlib)
```

---

## Instalação

```bash
# Crie e ative um ambiente virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Instale as dependências
pip install -r requirements.txt
```

---

## Execução

Execute sempre a partir da raiz do projeto (`trabalho-1/`):

```bash
python main.py
```

Para escolher qual exercício rodar, edite `main.py` e descomente a linha desejada:

```python
# ex1()   # Função algébrica
# ex2()   # Problema dos rádios
  ex3()   # 3-SAT  ← ativo por padrão
```

---

## Exemplos

### EX1 — Minimização/maximização de f(x) = cos(20x) − |x|/2 + x³/4

- Codificação binária, 16 bits, domínio x ∈ [−2, 2]
- Para trocar entre minimização e maximização, edite `MAXIMIZACAO` em `src/ex1.py`

```bash
python main.py   # (descomente ex1() em main.py)
```

### EX2 — Problema dos rádios com restrição

- Maximizar `30·ST + 40·LX` sujeito a `ST + 2·LX ≤ 40`
- Tratamento de restrição via penalidade: `fit = FOn + r·Hn`

```bash
python main.py   # (descomente ex2() em main.py)
```

### EX3 — 3-SAT

- 100 variáveis booleanas, 430 cláusulas
- Fitness = cláusulas satisfeitas / total de cláusulas
- Instância carregada de `data/ex3.cnf` (formato DIMACS CNF)

```bash
python main.py   # ex3() já está ativo por padrão
```

---

## Gráficos

Após a execução, os gráficos de convergência são salvos automaticamente em `output/<ex>/`:

```
output/ex3/convergencia_run_1.png
output/ex3/convergencia_run_2.png
...
```

Cada gráfico mostra:
- **Azul**: melhor fitness por geração
- **Vermelho**: média de fitness da população por geração

---

## Parâmetros do AG (config JSON)

| Campo       | Descrição                                      |
|-------------|------------------------------------------------|
| `cod`       | Tipo de codificação (`BIN`, `INT`, `INT_PERM`, `REAL`) |
| `pop`       | Tamanho da população                           |
| `dim`       | Dimensão do cromossomo (número de genes/bits)  |
| `gen`       | Número máximo de gerações                      |
| `run`       | Número de execuções independentes              |
| `bounds`    | Limites do espaço de busca (`min`, `max`)      |
| `pc`        | Probabilidade de crossover                     |
| `pm`        | Probabilidade de mutação                       |
| `crossover` | Tipo de crossover (`UM_PONTO`, `DOIS_PONTOS`, `UNIFORME`, `CX`, `PMX`) |
| `elitismo`  | Preservar o melhor indivíduo entre gerações    |

---

## Módulo `src/ae.py` — Operadores implementados

| Categoria  | Operador            | Codificação |
|------------|---------------------|-------------|
| Seleção    | Roleta sem reposição | Todas      |
| Crossover  | Um ponto            | BIN         |
| Crossover  | Dois pontos         | BIN         |
| Crossover  | Uniforme            | BIN         |
| Crossover  | CX (Cycle)          | INT_PERM    |
| Crossover  | PMX                 | INT_PERM    |
| Mutação    | Bit-flip            | BIN         |
| Mutação    | Swap                | INT_PERM    |
| Elitismo   | Substituição do pior | Todas      |
