from __future__ import annotations

import os

import matplotlib.pyplot as plt

from .ae import ResultadoRun


def gerar_grafico_convergencia(path: str, resultado: ResultadoRun) -> None:
    os.makedirs(path, exist_ok=True)

    nome_arquivo = os.path.join(path, f"convergencia_run_{resultado.run}.png")

    historico = resultado.historico
    geracoes = [h.geracao for h in historico]
    melhores = [h.melhor_fitness for h in historico]
    medias = [h.media_fitness for h in historico]

    min_fitness = min(min(melhores), min(medias))
    max_fitness = max(max(melhores), max(medias))

    margem_y = 1.0 if min_fitness == max_fitness else (max_fitness - min_fitness) * 0.1
    y_min = min_fitness - margem_y
    y_max = max_fitness + margem_y

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(geracoes, melhores, color="blue", label="Melhor fitness")
    ax.plot(geracoes, medias, color="red", label="Média fitness")

    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.set_xlim(0, max(geracoes) + 1)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    fig.tight_layout(pad=2.0)
    fig.savefig(nome_arquivo, dpi=100)
    plt.close(fig)

    print(f"Gráfico salvo em {nome_arquivo}")
