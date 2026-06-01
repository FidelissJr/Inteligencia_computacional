from __future__ import annotations

import random

from .ae import (
    Cromossomo,
    Config,
    Individuo,
    carregar_config,
    executar_com_config,
)
from .grafico import gerar_grafico_convergencia


# ---------------------------------------------------------------------------
# Cromossomo
# ---------------------------------------------------------------------------
# Codificação por permutação (INT_PERM): o cromossomo é uma permutação de
# [0, N-1] onde o gene da coluna i guarda a LINHA da rainha dessa coluna.
# Como toda permutação tem valores distintos, nunca há duas rainhas na mesma
# linha nem na mesma coluna por construção. Resta apenas a restrição diagonal.
# ---------------------------------------------------------------------------


def _gerar_individuo(n: int) -> Individuo:
    linhas = list(range(n))
    random.shuffle(linhas)
    return Individuo(Cromossomo("INT_PERM", linhas))


def _gerar_populacao_inicial(pop: int, n: int) -> list[Individuo]:
    return [_gerar_individuo(n) for _ in range(pop)]


# ---------------------------------------------------------------------------
# Função objetivo e fitness
# ---------------------------------------------------------------------------
# FO (a minimizar) = número de pares de rainhas que se atacam nas diagonais.
# Duas rainhas (i, perm[i]) e (j, perm[j]) se atacam na diagonal quando
# |i - j| == |perm[i] - perm[j]|. Contamos em O(N) agrupando por diagonal:
#   diagonal "/"  identificada por  (linha - coluna)
#   diagonal "\"  identificada por  (linha + coluna)
# Para cada diagonal com k rainhas, há C(k, 2) pares em colisão.
# ---------------------------------------------------------------------------


def _contar_colisoes(perm: list[int]) -> int:
    n = len(perm)
    diag_principal: dict[int, int] = {}
    diag_secundaria: dict[int, int] = {}

    for coluna in range(n):
        linha = perm[coluna]
        d1 = linha - coluna
        d2 = linha + coluna
        diag_principal[d1] = diag_principal.get(d1, 0) + 1
        diag_secundaria[d2] = diag_secundaria.get(d2, 0) + 1

    colisoes = 0
    for k in diag_principal.values():
        colisoes += k * (k - 1) // 2
    for k in diag_secundaria.values():
        colisoes += k * (k - 1) // 2
    return colisoes


def _max_pares(n: int) -> int:
    return n * (n - 1) // 2


def _fitness_nrainhas(cromossomo: Cromossomo) -> float:
    """Fitness normalizado em (0, 1]; 1.0 = solução sem colisões (maximização)."""
    perm = cromossomo.as_perm()
    colisoes = _contar_colisoes(perm)
    return 1.0 - colisoes / _max_pares(len(perm))


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------


def _avaliar_populacao_inicial(populacao: list[Individuo]) -> None:
    for ind in populacao:
        ind.fitness = _fitness_nrainhas(ind.cromossomo)


def _imprimir_individuo(rotulo: str, individuo: Individuo) -> None:
    perm = individuo.cromossomo.as_perm()
    colisoes = _contar_colisoes(perm)
    print(f"  {rotulo}: FO (colisões) = {colisoes} | fitness = {individuo.fitness:.6f}")


def _imprimir_resultado_final(n: int, individuo: Individuo) -> None:
    perm = individuo.cromossomo.as_perm()
    colisoes = _contar_colisoes(perm)
    infringiu = colisoes > 0

    print(f"\n  >>> Resultado final ({n} rainhas)")
    print(f"  Melhor indivíduo (linha por coluna) = {perm}")
    print(f"  Valor final da função objetivo (FO = colisões) = {colisoes}")
    print(f"  Fitness final = {individuo.fitness:.6f}")
    print(f"  Total de pares possíveis = {_max_pares(n)}")
    if infringiu:
        print("  Restrição (nenhuma rainha se ataca): INFRINGIDA "
              f"({colisoes} par(es) de rainhas em ataque)")
    else:
        print("  Restrição (nenhuma rainha se ataca): SATISFEITA — solução válida")


def _resolver_tabuleiro(config: Config, output_dir: str) -> None:
    n = config.dim
    print(f"\n========== TABULEIRO {n}x{n} ({n} rainhas) ==========")

    # 1. População inicial e avaliação
    populacao_inicial = _gerar_populacao_inicial(config.pop, n)
    _avaliar_populacao_inicial(populacao_inicial)

    # 2. Armazena melhor e pior indivíduos da população inicial
    melhor_inicial = max(populacao_inicial, key=lambda ind: ind.fitness).copy()
    pior_inicial = min(populacao_inicial, key=lambda ind: ind.fitness).copy()

    print(f"  População inicial ({config.pop} indivíduos) avaliada:")
    _imprimir_individuo("Melhor inicial", melhor_inicial)
    _imprimir_individuo("Pior inicial  ", pior_inicial)

    # 3. Evolução
    resultado = executar_com_config(config, _fitness_nrainhas)[0]

    # 4. Resultados finais
    _imprimir_resultado_final(n, resultado.melhor_individuo)
    gerar_grafico_convergencia(output_dir, resultado)


def ex4(
    config_path: str = "./config/config_ex4_8.json",
    output_dir: str = "./output/ex4",
) -> None:
    print("EX4 - Problema das N-Rainhas")
    print("Codificação: permutação (INT_PERM) — gene[coluna] = linha da rainha")
    print("FO (a minimizar) = nº de pares de rainhas em ataque diagonal")
    print("Fitness = 1 - colisões / C(N,2)   (maximização; 1.0 = solução)")
    print("Restrição: nenhuma rainha pode atacar outra (colisões = 0)")
    print(f"Config: {config_path}")

    config = carregar_config(config_path)
    _resolver_tabuleiro(config, f"{output_dir}/{config.dim}rainhas")
