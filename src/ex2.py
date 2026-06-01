from __future__ import annotations

from .ae import Cromossomo, executar
from .grafico import gerar_grafico_convergencia

ST_MIN, ST_MAX = 0.0, 24.0
LX_MIN, LX_MAX = 0.0, 16.0

BITS_ST = 5
BITS_LX = 5

R = -1.0  # coeficiente de penalidade


def _binario_para_decimal(bits: list[int]) -> int:
    result = 0
    for bit in bits:
        result = result * 2 + bit
    return result


def _ajustar_escala(bits: list[int], min_val: float, max_val: float) -> float:
    valor = _binario_para_decimal(bits)
    maior = (2 ** len(bits)) - 1
    return min_val + valor * (max_val - min_val) / maior


def _decodificar_st_lx(bits: list[int]) -> tuple[float, float]:
    if len(bits) != BITS_ST + BITS_LX:
        raise ValueError("Cromossomo inválido para EX2. Esperado: 10 bits")
    st = round(_ajustar_escala(bits[:BITS_ST], ST_MIN, ST_MAX))
    lx = round(_ajustar_escala(bits[BITS_ST:], LX_MIN, LX_MAX))
    return float(st), float(lx)


def _funcao_objetivo(st: float, lx: float) -> float:
    return 30.0 * st + 40.0 * lx


def _calcular_infracao(st: float, lx: float) -> float:
    h = st + 2.0 * lx
    return 0.0 if h <= 40.0 else h - 40.0


def _calcular_hn(st: float, lx: float) -> float:
    return max(0.0, (st + 2.0 * lx - 40.0) / 16.0)


def _fitness(cromossomo: Cromossomo) -> float:
    st, lx = _decodificar_st_lx(cromossomo.as_bin())
    fon = _funcao_objetivo(st, lx) / 1360.0
    hn = _calcular_hn(st, lx)
    return fon + R * hn


def _imprimir(cromossomo: Cromossomo) -> None:
    bits = cromossomo.as_bin()
    st, lx = _decodificar_st_lx(bits)
    fo = _funcao_objetivo(st, lx)
    fon = fo / 1360.0
    infracao = _calcular_infracao(st, lx)
    hn = _calcular_hn(st, lx)
    fitness = fon + R * hn

    print(f"Cromossomo = {bits}")
    print(f"ST = {st:.4f}")
    print(f"LX = {lx:.4f}")
    print(f"FO = {fo:.4f}")
    print(f"FOn = {fon:.4f}")
    print(f"Infração = {infracao:.4f}")
    print(f"Hn = {hn:.4f}")
    print(f"Fitness = {fitness:.4f}")


def ex2(
    config_path: str = "./config/config_ex2.json",
    output_dir: str = "./output/ex2",
) -> None:
    print("EX2 - Problema dos rádios")
    print("Fitness: fit = FOn + rHn")
    print("FOn = (30ST + 40LX) / 1360")
    print("Hn = max(0, (ST + 2LX - 40) / 16)")
    print(f"r = {R}")

    resultados = executar(config_path, _fitness)

    for resultado in resultados:
        print(f"\nRUN {resultado.run}")
        print("Melhor indivíduo da RUN:")
        _imprimir(resultado.melhor_individuo.cromossomo)
        gerar_grafico_convergencia(output_dir, resultado)
        print("--------------------------")
