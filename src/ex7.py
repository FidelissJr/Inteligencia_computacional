from __future__ import annotations

import math
import random

from .ae import (
    Config,
    Cromossomo,
    Individuo,
    carregar_config,
    executar_com_config,
)
from .ex4 import _contar_colisoes, _max_pares
from .grafico import gerar_grafico_convergencia


# ---------------------------------------------------------------------------
# Tabuleiro valorado
# ---------------------------------------------------------------------------
# As células são numeradas 1..N² em ordem de leitura (row-major):
#   valor_base(linha, coluna) = linha * N + coluna + 1   (linha, coluna 0-based)
# Em seguida transforma-se cada LINHA, alternando:
#   linha 1 (índice 0): sqrt      linha 2 (índice 1): log10
#   linha 3 (índice 2): sqrt      linha 4 (índice 3): log10  ...
# Ou seja: índice de linha par  -> sqrt ; índice de linha ímpar -> log10.
# ---------------------------------------------------------------------------


def _valor_celula(linha: int, coluna: int, n: int) -> float:
    base = linha * n + coluna + 1
    if linha % 2 == 0:
        return math.sqrt(base)
    return math.log10(base)


def _soma_valores(perm: list[int], n: int) -> float:
    # perm[coluna] = linha da rainha naquela coluna
    return sum(_valor_celula(perm[coluna], coluna, n) for coluna in range(n))


def _soma_max(n: int) -> float:
    """Cota superior da soma: maior valor possível de cada linha (coluna = N-1)."""
    return sum(_valor_celula(linha, n - 1, n) for linha in range(n))


def _imprimir_tabuleiro_valorado(n: int) -> None:
    print(f"  Tabuleiro valorado {n}x{n} (sqrt nas linhas ímpares, log10 nas pares):")
    for linha in range(n):
        op = "sqrt " if linha % 2 == 0 else "log10"
        celulas = " ".join(f"{_valor_celula(linha, c, n):6.3f}" for c in range(n))
        print(f"    L{linha + 1:>2} [{op}] {celulas}")


# ---------------------------------------------------------------------------
# Função objetivo e fitness
# ---------------------------------------------------------------------------
# FO (a maximizar) = soma dos valores das células ocupadas pelas rainhas.
# Restrição = nenhuma rainha pode atacar outra (colisões diagonais = 0).
# Fitness penalizado:  fitness = soma / soma_max - colisoes
# Como soma/soma_max ∈ (0, 1], cada colisão (>=1) leva o fitness a < 0,
# garantindo que toda solução válida supere qualquer solução inválida.
# ---------------------------------------------------------------------------


def _fitness_ex7(cromossomo: Cromossomo, n: int, soma_max: float) -> float:
    perm = cromossomo.as_perm()
    soma = _soma_valores(perm, n)
    colisoes = _contar_colisoes(perm)
    return soma / soma_max - colisoes


# ---------------------------------------------------------------------------
# População inicial
# ---------------------------------------------------------------------------


def _gerar_individuo(n: int) -> Individuo:
    linhas = list(range(n))
    random.shuffle(linhas)
    return Individuo(Cromossomo("INT_PERM", linhas))


def _gerar_populacao_inicial(pop: int, n: int) -> list[Individuo]:
    return [_gerar_individuo(n) for _ in range(pop)]


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------


def _imprimir_individuo(rotulo: str, individuo: Individuo, n: int) -> None:
    perm = individuo.cromossomo.as_perm()
    soma = _soma_valores(perm, n)
    colisoes = _contar_colisoes(perm)
    print(f"  {rotulo}: soma = {soma:.4f} | colisões = {colisoes} "
          f"| fitness = {individuo.fitness:.6f}")


def _imprimir_resultado_final(n: int, individuo: Individuo) -> None:
    perm = individuo.cromossomo.as_perm()
    soma = _soma_valores(perm, n)
    colisoes = _contar_colisoes(perm)
    infringiu = colisoes > 0

    print(f"\n  >>> Resultado final ({n} rainhas)")
    print(f"  Melhor indivíduo (linha por coluna) = {perm}")
    print(f"  Valor final da função objetivo (FO = soma dos valores) = {soma:.6f}")
    print(f"  Soma máxima teórica (cota superior) = {_soma_max(n):.6f}")
    print(f"  Fitness final = {individuo.fitness:.6f}")
    if infringiu:
        print("  Restrição (nenhuma rainha se ataca): INFRINGIDA "
              f"({colisoes} par(es) de rainhas em ataque)")
    else:
        print("  Restrição (nenhuma rainha se ataca): SATISFEITA — solução válida")


def ex7(
    config_path: str = "./config/config_ex7_8.json",
    output_dir: str = "./output/ex7",
) -> None:
    print("EX7 - N-Rainhas em tabuleiro valorado (maximização com restrição)")
    print("Codificação: permutação (INT_PERM) — gene[coluna] = linha da rainha")
    print("Tabuleiro: células 1..N² (row-major); sqrt nas linhas ímpares, log10 nas pares")
    print("FO (a maximizar) = soma dos valores das células ocupadas pelas rainhas")
    print("Fitness = soma/soma_max - colisões   (restrição: colisões = 0)")
    print(f"Config: {config_path}")

    config: Config = carregar_config(config_path)
    n = config.dim
    soma_max = _soma_max(n)

    print(f"\n========== TABULEIRO {n}x{n} ({n} rainhas) ==========")
    if n <= 8:
        _imprimir_tabuleiro_valorado(n)

    def fitness_fn(cromossomo: Cromossomo) -> float:
        return _fitness_ex7(cromossomo, n, soma_max)

    # População inicial e armazenamento de melhor/pior
    populacao_inicial = _gerar_populacao_inicial(config.pop, n)
    for ind in populacao_inicial:
        ind.fitness = fitness_fn(ind.cromossomo)
    melhor_inicial = max(populacao_inicial, key=lambda i: i.fitness).copy()
    pior_inicial = min(populacao_inicial, key=lambda i: i.fitness).copy()

    print(f"\n  População inicial ({config.pop} indivíduos) avaliada:")
    _imprimir_individuo("Melhor inicial", melhor_inicial, n)
    _imprimir_individuo("Pior inicial  ", pior_inicial, n)

    # Evolução (config.run execuções independentes)
    resultados = executar_com_config(config, fitness_fn)

    melhor_geral = max(
        (r.melhor_individuo for r in resultados), key=lambda i: i.fitness
    ).copy()

    print(f"\n  Resumo das {len(resultados)} run(s):")
    for resultado in resultados:
        _imprimir_individuo(f"RUN {resultado.run:>2}", resultado.melhor_individuo, n)
        gerar_grafico_convergencia(output_dir, resultado)

    print("\n  ===== MELHOR INDIVÍDUO ENTRE TODAS AS RUNS =====")
    _imprimir_resultado_final(n, melhor_geral)
