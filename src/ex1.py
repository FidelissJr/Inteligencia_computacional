from __future__ import annotations

import math

from .ae import Cromossomo, executar
from .grafico import gerar_grafico_convergencia

MIN = -2.0
MAX = 2.0

# False = minimização  |  True = maximização
MAXIMIZACAO = False


def _binario_para_decimal(bits: list[int]) -> int:
    result = 0
    for bit in bits:
        result = result * 2 + bit
    return result


def _decodificar_x(bits: list[int]) -> float:
    valor = _binario_para_decimal(bits)
    maior = (2 ** len(bits)) - 1
    return MIN + valor * (MAX - MIN) / maior


def _funcao_objetivo(x: float) -> float:
    return math.cos(20.0 * x) - abs(x) / 2.0 + x**3 / 4.0


def _fitness(cromossomo: Cromossomo) -> float:
    x = _decodificar_x(cromossomo.as_bin())
    fo = _funcao_objetivo(x)
    return 4.0 + fo if MAXIMIZACAO else 2.0 - fo


def _imprimir(cromossomo: Cromossomo) -> None:
    bits = cromossomo.as_bin()
    x = _decodificar_x(bits)
    fo = _funcao_objetivo(x)
    fitness = 4.0 + fo if MAXIMIZACAO else 2.0 - fo
    print(f"x = {x}")
    print(f"f(x) = {fo}")
    print(f"Fitness = {fitness}")


def ex1(
    config_path: str = "./config/config_ex1.json",
    output_dir: str = "./output/ex1",
) -> None:
    if MAXIMIZACAO:
        print("FITNESS PARA MAXIMIZAÇÃO: fit_Max = 4 + f(x)")
    else:
        print("FITNESS PARA MINIMIZAÇÃO: fit_Min = 2 - f(x)")

    resultados = executar(config_path, _fitness)

    for resultado in resultados:
        print(f"\nRUN {resultado.run}")
        _imprimir(resultado.melhor_individuo.cromossomo)
        gerar_grafico_convergencia(output_dir, resultado)
