from __future__ import annotations

"""
Genetic optimization scaffolding (optional).

Evolves numeric strategy parameters to maximize a provided scoring function.
This is intentionally lightweight and not wired into production.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List
import random
import math


Params = Dict[str, float]
Bounds = Dict[str, Tuple[float, float]]
ScoreFn = Callable[[Params], float]


@dataclass
class GAConfig:
    population_size: int = 30
    elite_frac: float = 0.1
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    generations: int = 20
    seed: int = 42
    sigma: float = 0.1  # relative mutation size


def _init_individual(bounds: Bounds) -> Params:
    return {k: random.uniform(lo, hi) for k, (lo, hi) in bounds.items()}


def _clip(params: Params, bounds: Bounds) -> Params:
    out = {}
    for k, v in params.items():
        lo, hi = bounds.get(k, (-math.inf, math.inf))
        out[k] = min(max(v, lo), hi)
    return out


def _crossover(a: Params, b: Params, rate: float) -> Tuple[Params, Params]:
    if random.random() > rate:
        return a.copy(), b.copy()
    child1, child2 = {}, {}
    for k in a.keys():
        if random.random() < 0.5:
            child1[k] = a[k]
            child2[k] = b[k]
        else:
            child1[k] = b[k]
            child2[k] = a[k]
    return child1, child2


def _mutate(x: Params, bounds: Bounds, sigma: float, rate: float) -> Params:
    out = x.copy()
    for k in out.keys():
        if random.random() < rate:
            span = max(1e-9, bounds[k][1] - bounds[k][0])
            out[k] += random.gauss(0.0, sigma * span)
    return _clip(out, bounds)


def run_ga(bounds: Bounds, score_fn: ScoreFn, cfg: GAConfig | None = None) -> Tuple[Params, float, List[Tuple[int, float]]]:
    cfg = cfg or GAConfig()
    random.seed(cfg.seed)
    pop: List[Params] = [_init_individual(bounds) for _ in range(cfg.population_size)]
    history: List[Tuple[int, float]] = []
    elite_n = max(1, int(cfg.elite_frac * cfg.population_size))

    best_params: Params = pop[0].copy()
    best_score: float = float('-inf')

    for gen in range(cfg.generations):
        scored = [(p, float(score_fn(p))) for p in pop]
        scored.sort(key=lambda t: t[1], reverse=True)
        if scored[0][1] > best_score:
            best_params, best_score = scored[0]
        history.append((gen, scored[0][1]))

        # Elitism
        elites = [s[0] for s in scored[:elite_n]]

        # Tournament selection
        def select() -> Params:
            a, b = random.choice(scored[: max(2, elite_n * 2)]), random.choice(scored[elite_n:])
            return a[0] if a[1] >= b[1] else b[0]

        new_pop: List[Params] = []
        # Preserve elites
        new_pop.extend(e.copy() for e in elites)
        # Fill remainder with crossover + mutation
        while len(new_pop) < cfg.population_size:
            p1, p2 = select(), select()
            c1, c2 = _crossover(p1, p2, cfg.crossover_rate)
            c1 = _mutate(c1, bounds, cfg.sigma, cfg.mutation_rate)
            c2 = _mutate(c2, bounds, cfg.sigma, cfg.mutation_rate)
            new_pop.append(c1)
            if len(new_pop) < cfg.population_size:
                new_pop.append(c2)
        pop = new_pop

    return best_params, best_score, history

