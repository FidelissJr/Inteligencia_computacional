from __future__ import annotations

from dataclasses import dataclass

from .ae import Cromossomo, executar
from .grafico import gerar_grafico_convergencia


@dataclass
class InstanciaSat:
    qtd_variaveis: int
    qtd_clausulas: int
    clausulas: list[list[int]]


def _carregar_instancia_sat(path: str) -> InstanciaSat:
    with open(path, "r") as f:
        conteudo = f.read()

    qtd_variaveis = 0
    qtd_clausulas = 0
    clausulas: list[list[int]] = []

    for linha in conteudo.splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("c"):
            continue
        if linha.startswith("%"):
            break
        if linha.startswith("p"):
            partes = linha.split()
            if len(partes) < 4:
                raise ValueError(f"Linha de cabeçalho inválida: {linha}")
            qtd_variaveis = int(partes[2])
            qtd_clausulas = int(partes[3])
            continue

        clause: list[int] = []
        for token in linha.split():
            lit = int(token)
            if lit == 0:
                break
            clause.append(lit)

        if clause:
            clausulas.append(clause)

    if len(clausulas) != qtd_clausulas:
        raise ValueError(
            f"Quantidade de cláusulas lida ({len(clausulas)}) diferente "
            f"da informada no arquivo ({qtd_clausulas})"
        )

    return InstanciaSat(
        qtd_variaveis=qtd_variaveis,
        qtd_clausulas=qtd_clausulas,
        clausulas=clausulas,
    )


def _literal_satisfeito(literal: int, bits: list[int]) -> bool:
    idx = abs(literal) - 1
    if idx >= len(bits):
        raise ValueError(f"Literal {literal} referencia variável fora do cromossomo")
    valor = bits[idx] == 1
    return valor if literal > 0 else not valor


def _clausula_satisfeita(clausula: list[int], bits: list[int]) -> bool:
    return any(_literal_satisfeito(lit, bits) for lit in clausula)


def _contar_clausulas_satisfeitas(bits: list[int], instancia: InstanciaSat) -> int:
    return sum(1 for c in instancia.clausulas if _clausula_satisfeita(c, bits))


def _fitness_3sat(cromossomo: Cromossomo, instancia: InstanciaSat) -> float:
    bits = cromossomo.as_bin()
    if len(bits) != instancia.qtd_variaveis:
        raise ValueError(
            f"Cromossomo inválido. Esperado {instancia.qtd_variaveis} bits, "
            f"recebido {len(bits)}"
        )
    return _contar_clausulas_satisfeitas(bits, instancia) / instancia.qtd_clausulas


def _imprimir(cromossomo: Cromossomo, instancia: InstanciaSat) -> None:
    bits = cromossomo.as_bin()
    fo = _contar_clausulas_satisfeitas(bits, instancia)
    fitness = fo / instancia.qtd_clausulas
    nao_sat = instancia.qtd_clausulas - fo

    print(f"Cromossomo = {bits}")
    print(f"FO = {fo}")
    print(f"Total de cláusulas = {instancia.qtd_clausulas}")
    print(f"Cláusulas não satisfeitas = {nao_sat}")
    print(f"Fitness = {fitness:.6f}")
    print("Solução SAT? " + ("Sim" if fo == instancia.qtd_clausulas else "Não"))


def ex3(
    config_path: str = "./config/config_ex3.json",
    data_path: str = "./data/ex3.cnf",
    output_dir: str = "./output/ex3",
) -> None:
    print("EX3 - 3-SAT")
    print("Codificação binária")
    print("FO = quantidade de cláusulas satisfeitas")
    print("Fitness = FO / total_clausulas")
    print("Objetivo: maximização")

    instancia = _carregar_instancia_sat(data_path)
    print(f"Variáveis: {instancia.qtd_variaveis}")
    print(f"Cláusulas: {instancia.qtd_clausulas}")

    resultados = executar(
        config_path,
        lambda cromossomo: _fitness_3sat(cromossomo, instancia),
    )

    for resultado in resultados:
        print(f"\nRUN {resultado.run}")
        print("Melhor indivíduo da RUN:")
        _imprimir(resultado.melhor_individuo.cromossomo, instancia)
        gerar_grafico_convergencia(output_dir, resultado)
        print("--------------------------")
