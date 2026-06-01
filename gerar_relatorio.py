"""
Executa todos os experimentos e gera figuras + JSON de resultados para o relatório.
Execute da raiz do projeto (trabalho-1/).
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ae import executar, Cromossomo, ResultadoRun

FIG_DIR = "./relatorio/figuras"
os.makedirs(FIG_DIR, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _b2d(bits: list[int]) -> int:
    r = 0
    for b in bits:
        r = r * 2 + b
    return r


def _scale(bits: list[int], mn: float, mx: float) -> float:
    return mn + _b2d(bits) * (mx - mn) / ((2 ** len(bits)) - 1)


# ─── EX1 ─────────────────────────────────────────────────────────────────────

def _f(x: float) -> float:
    return math.cos(20.0 * x) - abs(x) / 2.0 + x ** 3 / 4.0


def _decode_x(bits: list[int]) -> float:
    return _scale(bits, -2.0, 2.0)


def fitness_ex1_max(c: Cromossomo) -> float:
    return 4.0 + _f(_decode_x(c.as_bin()))


def fitness_ex1_min(c: Cromossomo) -> float:
    return 2.0 - _f(_decode_x(c.as_bin()))


# ─── EX2 ─────────────────────────────────────────────────────────────────────

def _decode_ex2(bits: list[int]) -> tuple[float, float]:
    return float(round(_scale(bits[:5], 0.0, 24.0))), float(round(_scale(bits[5:], 0.0, 16.0)))


def fitness_ex2(c: Cromossomo) -> float:
    st, lx = _decode_ex2(c.as_bin())
    fon = (30.0 * st + 40.0 * lx) / 1360.0
    hn = max(0.0, (st + 2.0 * lx - 40.0) / 16.0)
    return fon - hn


# ─── EX3 ─────────────────────────────────────────────────────────────────────

def _load_sat(path: str):
    with open(path) as f:
        text = f.read()
    num_vars = num_clauses = 0
    clauses: list[list[int]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("%"):
            break
        if line.startswith("p"):
            parts = line.split()
            num_vars, num_clauses = int(parts[2]), int(parts[3])
            continue
        lits = []
        for tok in line.split():
            v = int(tok)
            if v == 0:
                break
            lits.append(v)
        if lits:
            clauses.append(lits)
    return num_vars, num_clauses, clauses


def _make_sat_fitness(clauses: list[list[int]], num_clauses: int):
    def fit(c: Cromossomo) -> float:
        bits = c.as_bin()
        sat = sum(
            any((bits[abs(lit) - 1] == 1) == (lit > 0) for lit in clause)
            for clause in clauses
        )
        return sat / num_clauses
    return fit


# ─── Plots ───────────────────────────────────────────────────────────────────

def _convergence_overlay(resultados: list[ResultadoRun], title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    n_gens = len(resultados[0].historico)
    gens = list(range(n_gens))

    for r in resultados:
        ax.plot(gens, [h.melhor_fitness for h in r.historico],
                color="steelblue", alpha=0.22, linewidth=0.7)
        ax.plot(gens, [h.media_fitness for h in r.historico],
                color="tomato", alpha=0.16, linewidth=0.6)

    mean_best = [sum(r.historico[g].melhor_fitness for r in resultados) / len(resultados) for g in range(n_gens)]
    mean_avg  = [sum(r.historico[g].media_fitness  for r in resultados) / len(resultados) for g in range(n_gens)]

    ax.plot(gens, mean_best, color="steelblue", linewidth=2.0, label="Melhor fitness (média)")
    ax.plot(gens, mean_avg,  color="tomato",    linewidth=2.0, label="Média de fitness (média)")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def _boxplot_combined(data: dict[str, list[float]], fname: str) -> None:
    labels = list(data.keys())
    values = [data[k] for k in labels]
    fig, axes = plt.subplots(1, len(labels), figsize=(10, 4))
    bp_kw = dict(patch_artist=True,
                 boxprops=dict(facecolor="lightsteelblue"),
                 medianprops=dict(color="navy", linewidth=2.2),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker="o", markersize=4, markerfacecolor="gray"))
    for ax, label, vals in zip(axes, labels, values):
        ax.boxplot([vals], tick_labels=[""], **bp_kw)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("Fitness")
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        # auto-scale with small margin
        lo, hi = min(vals), max(vals)
        margin = max((hi - lo) * 0.5, 0.005)
        ax.set_ylim(lo - margin, hi + margin)
    fig.suptitle("Distribuição do Melhor Fitness por Experimento (10 execuções)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def _stats(v: list[float]) -> dict:
    a = np.array(v)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


# ─── Main ────────────────────────────────────────────────────────────────────

print("=" * 64)
print("  EX1 — Maximização")
print("=" * 64)
res1max = executar("./config/config_ex1.json", fitness_ex1_max)
ex1max_rows = []
for r in res1max:
    bits = r.melhor_individuo.cromossomo.as_bin()
    x = _decode_x(bits); fx = _f(x)
    ex1max_rows.append({"run": r.run, "x": x, "fx": fx, "fitness": r.melhor_individuo.fitness})
    print(f"  Run {r.run:2d}  x={x:+.6f}  f(x)={fx:+.6f}  fit={r.melhor_individuo.fitness:.6f}")
_convergence_overlay(res1max, "EX1 — Maximização (10 execuções)", "ex1_max_conv.png")

print("\n" + "=" * 64)
print("  EX1 — Minimização")
print("=" * 64)
res1min = executar("./config/config_ex1.json", fitness_ex1_min)
ex1min_rows = []
for r in res1min:
    bits = r.melhor_individuo.cromossomo.as_bin()
    x = _decode_x(bits); fx = _f(x)
    ex1min_rows.append({"run": r.run, "x": x, "fx": fx, "fitness": r.melhor_individuo.fitness})
    print(f"  Run {r.run:2d}  x={x:+.6f}  f(x)={fx:+.6f}  fit={r.melhor_individuo.fitness:.6f}")
_convergence_overlay(res1min, "EX1 — Minimização (10 execuções)", "ex1_min_conv.png")

print("\n" + "=" * 64)
print("  EX2 — Problema dos Rádios")
print("=" * 64)
res2 = executar("./config/config_ex2.json", fitness_ex2)
ex2_rows = []
for r in res2:
    st, lx = _decode_ex2(r.melhor_individuo.cromossomo.as_bin())
    fo = 30 * st + 40 * lx
    inf_ = max(0.0, st + 2 * lx - 40.0)
    hn   = max(0.0, (st + 2 * lx - 40.0) / 16.0)
    ex2_rows.append({"run": r.run, "st": st, "lx": lx, "fo": fo,
                     "infracao": inf_, "hn": hn, "fitness": r.melhor_individuo.fitness})
    print(f"  Run {r.run:2d}  ST={st:.0f}  LX={lx:.0f}  FO={fo:.0f}  inf={inf_:.0f}  fit={r.melhor_individuo.fitness:.4f}")
_convergence_overlay(res2, "EX2 — Problema dos Rádios (10 execuções)", "ex2_conv.png")

print("\n" + "=" * 64)
print("  EX3 — 3-SAT")
print("=" * 64)
num_vars, num_clauses, clauses = _load_sat("./data/ex3.cnf")
print(f"  Instância: {num_vars} variáveis, {num_clauses} cláusulas")
res3 = executar("./config/config_ex3.json", _make_sat_fitness(clauses, num_clauses))
ex3_rows = []
for r in res3:
    sat = int(round(r.melhor_individuo.fitness * num_clauses))
    ex3_rows.append({"run": r.run, "satisfeitas": sat, "nao_sat": num_clauses - sat,
                     "fitness": r.melhor_individuo.fitness})
    print(f"  Run {r.run:2d}  sat={sat}  nao_sat={num_clauses-sat}  fit={r.melhor_individuo.fitness:.6f}")
_convergence_overlay(res3, "EX3 — 3-SAT (10 execuções)", "ex3_conv.png")

print("\n[Box plots]")
_boxplot_combined({
    "EX1\nMaximização": [r["fitness"] for r in ex1max_rows],
    "EX1\nMinimização": [r["fitness"] for r in ex1min_rows],
    "EX2\nRádios":      [r["fitness"] for r in ex2_rows],
    "EX3\n3-SAT":       [r["fitness"] for r in ex3_rows],
}, "boxplot_geral.png")

# Estatísticas
stats = {
    "ex1_max": _stats([r["fitness"] for r in ex1max_rows]),
    "ex1_min": _stats([r["fitness"] for r in ex1min_rows]),
    "ex2":     _stats([r["fitness"] for r in ex2_rows]),
    "ex3":     _stats([r["fitness"] for r in ex3_rows]),
}
print("\n--- Estatísticas ---")
for k, s in stats.items():
    print(f"  {k}: min={s['min']:.6f}  max={s['max']:.6f}  média={s['mean']:.6f}  dp={s['std']:.6f}")

# Salva JSON
dados = {
    "ex1_max": ex1max_rows, "ex1_min": ex1min_rows,
    "ex2": ex2_rows, "ex3": ex3_rows, "stats": stats,
    "num_clauses": num_clauses, "num_vars": num_vars,
}
json_path = "./relatorio/dados_resultados.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(dados, f, indent=2, ensure_ascii=False)
print(f"\nResultados salvos em {json_path}")
print("Concluído!")
