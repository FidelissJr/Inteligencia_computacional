from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Bounds:
    min: float
    max: float


@dataclass
class Config:
    cod: str
    pop: int
    dim: int
    geracoes: int
    run: int
    bounds: Bounds
    pc: float
    pm: float
    crossover: str
    elitismo: bool


class Cromossomo:
    """Cromossomo com tipo de codificação e dados genéticos."""

    def __init__(self, cod: str, data: list):
        self.cod = cod
        self.data = data

    def as_bin(self) -> list[int]:
        if self.cod != "BIN":
            raise ValueError("Cromossomo não é BIN")
        return self.data

    def as_int(self) -> list[int]:
        if self.cod != "INT":
            raise ValueError("Cromossomo não é INT")
        return self.data

    def as_perm(self) -> list[int]:
        if self.cod != "INT_PERM":
            raise ValueError("Cromossomo não é INT_PERM")
        return self.data

    def as_real(self) -> list[float]:
        if self.cod != "REAL":
            raise ValueError("Cromossomo não é REAL")
        return self.data

    def copy(self) -> Cromossomo:
        return Cromossomo(self.cod, self.data.copy())

    def __repr__(self) -> str:
        return f"Cromossomo({self.cod}, {self.data})"


class Individuo:
    """Indivíduo com cromossomo e valor de fitness."""

    def __init__(self, cromossomo: Cromossomo):
        self.cromossomo = cromossomo
        self._fitness: Optional[float] = None

    @property
    def fitness(self) -> float:
        if self._fitness is None:
            raise ValueError("Indivíduo ainda não foi avaliado")
        return self._fitness

    @fitness.setter
    def fitness(self, value: Optional[float]) -> None:
        self._fitness = value

    def copy(self) -> Individuo:
        ind = Individuo(self.cromossomo.copy())
        ind._fitness = self._fitness
        return ind

    def __repr__(self) -> str:
        return f"Individuo(fitness={self._fitness})"


@dataclass
class HistoricoGeracao:
    geracao: int
    melhor_fitness: float
    media_fitness: float


@dataclass
class ResultadoRun:
    run: int
    melhor_individuo: Individuo
    historico: list[HistoricoGeracao]


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def carregar_config(path: str) -> Config:
    with open(path, "r") as f:
        data = json.load(f)

    bounds = Bounds(min=data["bounds"]["min"], max=data["bounds"]["max"])

    return Config(
        cod=data["cod"],
        pop=data["pop"],
        dim=data["dim"],
        geracoes=data["gen"],
        run=data["run"],
        bounds=bounds,
        pc=data["pc"],
        pm=data["pm"],
        crossover=data["crossover"],
        elitismo=data["elitismo"],
    )


def _validar_config(config: Config) -> None:
    if not (0.0 <= config.pc <= 1.0):
        raise ValueError("PC deve estar entre 0 e 1")
    if not (0.0 <= config.pm <= 1.0):
        raise ValueError("PM deve estar entre 0 e 1")
    if config.cod not in ("BIN", "INT", "INT_PERM", "REAL"):
        raise ValueError("COD inválido. Use BIN, INT, INT_PERM ou REAL")
    if config.cod == "BIN" and config.crossover not in ("UM_PONTO", "DOIS_PONTOS", "UNIFORME"):
        raise ValueError("Crossover inválido para BIN. Use UM_PONTO, DOIS_PONTOS ou UNIFORME")
    if config.cod == "INT_PERM" and config.crossover not in ("CX", "PMX"):
        raise ValueError("Crossover inválido para INT_PERM. Use CX ou PMX")


# ---------------------------------------------------------------------------
# Geração de população inicial
# ---------------------------------------------------------------------------

def _gerar_populacao(config: Config) -> list[Individuo]:
    if config.cod == "BIN":
        return _gerar_binario(config.pop, config.dim)
    if config.cod == "INT":
        return _gerar_inteiro(config.pop, config.dim, int(config.bounds.min), int(config.bounds.max))
    if config.cod == "INT_PERM":
        return _gerar_perm(config.pop, config.dim)
    if config.cod == "REAL":
        return _gerar_real(config.pop, config.dim, config.bounds.min, config.bounds.max)
    raise ValueError("COD inválido")


def _gerar_binario(pop: int, dim: int) -> list[Individuo]:
    return [
        Individuo(Cromossomo("BIN", [random.randint(0, 1) for _ in range(dim)]))
        for _ in range(pop)
    ]


def _gerar_inteiro(pop: int, dim: int, min_val: int, max_val: int) -> list[Individuo]:
    return [
        Individuo(Cromossomo("INT", [random.randint(min_val, max_val) for _ in range(dim)]))
        for _ in range(pop)
    ]


def _gerar_perm(pop: int, dim: int) -> list[Individuo]:
    individuos = []
    for _ in range(pop):
        data = list(range(dim))
        random.shuffle(data)
        individuos.append(Individuo(Cromossomo("INT_PERM", data)))
    return individuos


def _gerar_real(pop: int, dim: int, min_val: float, max_val: float) -> list[Individuo]:
    return [
        Individuo(Cromossomo("REAL", [random.uniform(min_val, max_val) for _ in range(dim)]))
        for _ in range(pop)
    ]


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def _avaliar_populacao(
    populacao: list[Individuo],
    fitness_fn: Callable[[Cromossomo], float],
) -> None:
    for ind in populacao:
        ind.fitness = fitness_fn(ind.cromossomo)


def _calcular_media_fitness(populacao: list[Individuo]) -> float:
    return sum(ind.fitness for ind in populacao) / len(populacao)


def _obter_melhor_individuo(populacao: list[Individuo]) -> Individuo:
    return max(populacao, key=lambda ind: ind.fitness)


def _obter_indice_pior_individuo(populacao: list[Individuo]) -> int:
    return min(range(len(populacao)), key=lambda i: populacao[i].fitness)


def _aplicar_elitismo(populacao: list[Individuo], elite: Individuo) -> None:
    populacao[_obter_indice_pior_individuo(populacao)] = elite


# ---------------------------------------------------------------------------
# Seleção por roleta sem reposição
# ---------------------------------------------------------------------------

def _calcular_pesos_roleta(populacao: list[Individuo]) -> list[float]:
    EPS = 1e-9
    menor = min(ind.fitness for ind in populacao)
    if menor <= 0.0:
        return [ind.fitness - menor + EPS for ind in populacao]
    return [ind.fitness + EPS for ind in populacao]


def _sortear_indice_roleta(pesos: list[float], ignorar: Optional[int]) -> int:
    total = sum(p for i, p in enumerate(pesos) if i != ignorar)

    if total <= 0.0:
        validos = [i for i in range(len(pesos)) if i != ignorar]
        return random.choice(validos)

    sorteio = random.uniform(0.0, total)
    for i, peso in enumerate(pesos):
        if i == ignorar:
            continue
        sorteio -= peso
        if sorteio <= 0.0:
            return i

    # fallback: último índice válido
    for i in range(len(pesos) - 1, -1, -1):
        if i != ignorar:
            return i
    raise ValueError("Nenhum índice válido encontrado")


def _roleta_sem_reposicao(populacao: list[Individuo]) -> list[Individuo]:
    if not populacao:
        raise ValueError("População vazia")
    if len(populacao) == 1:
        return [populacao[0].copy()]

    pesos = _calcular_pesos_roleta(populacao)
    intermediaria: list[Individuo] = []

    while len(intermediaria) < len(populacao):
        idx1 = _sortear_indice_roleta(pesos, None)
        idx2 = _sortear_indice_roleta(pesos, idx1)
        intermediaria.append(populacao[idx1].copy())
        if len(intermediaria) < len(populacao):
            intermediaria.append(populacao[idx2].copy())

    return intermediaria


# ---------------------------------------------------------------------------
# Mutação
# ---------------------------------------------------------------------------

def _mutacao_bit_flip(bits: list[int], pm: float) -> None:
    for i in range(len(bits)):
        if random.random() < pm:
            bits[i] = 1 - bits[i]


def _mutacao_swap(valores: list[int], pm: float) -> None:
    if len(valores) < 2:
        return
    if random.random() < pm:
        i = random.randrange(len(valores))
        j = random.randrange(len(valores))
        while i == j:
            j = random.randrange(len(valores))
        valores[i], valores[j] = valores[j], valores[i]


def _aplicar_mutacao(populacao: list[Individuo], config: Config) -> None:
    for ind in populacao:
        if ind.cromossomo.cod == "BIN":
            _mutacao_bit_flip(ind.cromossomo.data, config.pm)
            ind.fitness = None
        elif ind.cromossomo.cod == "INT_PERM":
            _mutacao_swap(ind.cromossomo.data, config.pm)
            ind.fitness = None


# ---------------------------------------------------------------------------
# Crossover binário
# ---------------------------------------------------------------------------

def _crossover_bin_um_ponto(
    pai1: list[int], pai2: list[int]
) -> tuple[list[int], list[int]]:
    n = len(pai1)
    if n < 2:
        return pai1.copy(), pai2.copy()
    ponto = random.randint(1, n - 1)
    return pai1[:ponto] + pai2[ponto:], pai2[:ponto] + pai1[ponto:]


def _crossover_bin_dois_pontos(
    pai1: list[int], pai2: list[int]
) -> tuple[list[int], list[int]]:
    n = len(pai1)
    if n < 3:
        return _crossover_bin_um_ponto(pai1, pai2)

    p1 = random.randrange(n)
    p2 = random.randrange(n)
    while p1 == p2:
        p2 = random.randrange(n)
    if p1 > p2:
        p1, p2 = p2, p1

    filho1 = pai1.copy()
    filho2 = pai2.copy()
    for i in range(p1, p2):
        filho1[i] = pai2[i]
        filho2[i] = pai1[i]
    return filho1, filho2


def _crossover_bin_uniforme(
    pai1: list[int], pai2: list[int]
) -> tuple[list[int], list[int]]:
    filho1, filho2 = [], []
    for a, b in zip(pai1, pai2):
        if random.random() < 0.5:
            filho1.append(a)
            filho2.append(b)
        else:
            filho1.append(b)
            filho2.append(a)
    return filho1, filho2


# ---------------------------------------------------------------------------
# Crossover de permutação
# ---------------------------------------------------------------------------

def _crossover_perm_cx(
    pai1: list[int], pai2: list[int]
) -> tuple[list[int], list[int]]:
    n = len(pai1)
    if n == 0:
        return [], []

    filho1: list[Optional[int]] = [None] * n
    filho2: list[Optional[int]] = [None] * n
    visitado = [False] * n
    ciclo = 0

    while True:
        inicio = next((i for i in range(n) if not visitado[i]), None)
        if inicio is None:
            break

        idx = inicio
        while True:
            visitado[idx] = True
            if ciclo % 2 == 0:
                filho1[idx] = pai1[idx]
                filho2[idx] = pai2[idx]
            else:
                filho1[idx] = pai2[idx]
                filho2[idx] = pai1[idx]

            valor_no_pai2 = pai2[idx]
            idx = pai1.index(valor_no_pai2)
            if visitado[idx]:
                break

        ciclo += 1

    return filho1, filho2  # type: ignore[return-value]


def _pmx_gerar_filho(
    pai_base: list[int],
    pai_complementar: list[int],
    inicio: int,
    fim: int,
) -> list[int]:
    n = len(pai_base)
    VAZIO = None
    filho: list[Optional[int]] = [VAZIO] * n

    # 1. Copia trecho central do pai_base
    for i in range(inicio, fim + 1):
        filho[i] = pai_base[i]

    # 2. Insere genes do pai_complementar via mapeamento PMX
    for i in range(inicio, fim + 1):
        gene = pai_complementar[i]
        if gene in filho:
            continue

        pos = i
        while True:
            gene_mapeado = pai_base[pos]
            pos = pai_complementar.index(gene_mapeado)
            if filho[pos] is VAZIO:
                filho[pos] = gene
                break

    # 3. Preenche posições restantes com pai_complementar
    for i in range(n):
        if filho[i] is VAZIO:
            filho[i] = pai_complementar[i]

    return filho  # type: ignore[return-value]


def _crossover_perm_pmx(
    pai1: list[int], pai2: list[int]
) -> tuple[list[int], list[int]]:
    n = len(pai1)
    if n < 2:
        return pai1.copy(), pai2.copy()

    p1 = random.randrange(n)
    p2 = random.randrange(n)
    while p1 == p2:
        p2 = random.randrange(n)
    if p1 > p2:
        p1, p2 = p2, p1

    return _pmx_gerar_filho(pai1, pai2, p1, p2), _pmx_gerar_filho(pai2, pai1, p1, p2)


# ---------------------------------------------------------------------------
# Aplicação de crossover na população
# ---------------------------------------------------------------------------

def _crossover_individuos(
    pai1: Individuo, pai2: Individuo, config: Config
) -> tuple[Individuo, Individuo]:
    cod = pai1.cromossomo.cod

    if cod == "BIN":
        b1, b2 = pai1.cromossomo.data, pai2.cromossomo.data
        if config.crossover == "UM_PONTO":
            f1d, f2d = _crossover_bin_um_ponto(b1, b2)
        elif config.crossover == "DOIS_PONTOS":
            f1d, f2d = _crossover_bin_dois_pontos(b1, b2)
        elif config.crossover == "UNIFORME":
            f1d, f2d = _crossover_bin_uniforme(b1, b2)
        else:
            raise ValueError(f"Crossover BIN inválido: {config.crossover}")
        return Individuo(Cromossomo("BIN", f1d)), Individuo(Cromossomo("BIN", f2d))

    if cod == "INT_PERM":
        p1, p2 = pai1.cromossomo.data, pai2.cromossomo.data
        if config.crossover == "CX":
            f1d, f2d = _crossover_perm_cx(p1, p2)
        elif config.crossover == "PMX":
            f1d, f2d = _crossover_perm_pmx(p1, p2)
        else:
            raise ValueError(f"Crossover INT_PERM inválido: {config.crossover}")
        return Individuo(Cromossomo("INT_PERM", f1d)), Individuo(Cromossomo("INT_PERM", f2d))

    raise ValueError("Crossover ainda não implementado para essa representação")


def _aplicar_crossover(populacao: list[Individuo], config: Config) -> list[Individuo]:
    nova: list[Individuo] = []
    i = 0

    while i < len(populacao):
        if i + 1 >= len(populacao):
            ultimo = populacao[i].copy()
            ultimo.fitness = None
            nova.append(ultimo)
            break

        pai1, pai2 = populacao[i], populacao[i + 1]

        if random.random() < config.pc:
            filho1, filho2 = _crossover_individuos(pai1, pai2, config)
        else:
            filho1, filho2 = pai1.copy(), pai2.copy()

        filho1.fitness = None
        filho2.fitness = None

        nova.append(filho1)
        if len(nova) < len(populacao):
            nova.append(filho2)

        i += 2

    return nova


# ---------------------------------------------------------------------------
# Loop principal do AG
# ---------------------------------------------------------------------------

def executar(
    config_path: str,
    fitness_fn: Callable[[Cromossomo], float],
) -> list[ResultadoRun]:
    config = carregar_config(config_path)
    return executar_com_config(config, fitness_fn)


def executar_com_config(
    config: Config,
    fitness_fn: Callable[[Cromossomo], float],
) -> list[ResultadoRun]:
    _validar_config(config)

    resultados: list[ResultadoRun] = []

    for run in range(config.run):
        t = 0
        historico: list[HistoricoGeracao] = []

        # P(0): inicializa e avalia
        populacao = _gerar_populacao(config)
        _avaliar_populacao(populacao, fitness_fn)

        historico.append(HistoricoGeracao(
            geracao=t,
            melhor_fitness=_obter_melhor_individuo(populacao).fitness,
            media_fitness=_calcular_media_fitness(populacao),
        ))

        melhor_individuo = _obter_melhor_individuo(populacao).copy()

        while t < config.geracoes:
            t += 1

            elite = _obter_melhor_individuo(populacao).copy()

            # 1. Seleção
            pop_intermediaria = _roleta_sem_reposicao(populacao)

            # 2. Crossover
            nova_pop = _aplicar_crossover(pop_intermediaria, config)

            # 3. Mutação
            _aplicar_mutacao(nova_pop, config)

            # 4. Avaliação
            _avaliar_populacao(nova_pop, fitness_fn)

            # 5. Elitismo
            if config.elitismo:
                _aplicar_elitismo(nova_pop, elite)

            melhor_ger = _obter_melhor_individuo(nova_pop)
            media_ger = _calcular_media_fitness(nova_pop)

            historico.append(HistoricoGeracao(
                geracao=t,
                melhor_fitness=melhor_ger.fitness,
                media_fitness=media_ger,
            ))

            if melhor_ger.fitness > melhor_individuo.fitness:
                melhor_individuo = melhor_ger.copy()

            populacao = nova_pop

        resultados.append(ResultadoRun(
            run=run + 1,
            melhor_individuo=melhor_individuo,
            historico=historico,
        ))

    return resultados
