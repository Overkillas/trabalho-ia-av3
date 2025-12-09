import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 




def f1(x):
    x1, x2 = x
    return x1**2 + x2**2


bounds1 = [(-100, 100), (-100, 100)]
goal1 = "min"


def f2(x):
    x1, x2 = x
    # f(x1, x2) = exp(-(x1^2 + x2^2)) + 2 * exp(-(x1-1.7)^2 - (x2-1.7)^2)
    return math.exp(-(x1**2 + x2**2)) + 2 * math.exp(-((x1 - 1.7) ** 2 + (x2 - 1.7) ** 2))


bounds2 = [(-2, 4), (-2, 5)]
goal2 = "max"


def f3(x):
    x1, x2 = x
    term1 = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -math.exp(0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2)))
    return term1 + term2 + 20 + math.e


bounds3 = [(-8, 8), (-8, 8)]
goal3 = "min"


def f4(x):
    x1, x2 = x
    return (x1**2 - 10 * math.cos(2 * math.pi * x1) + 10) + (x2**2 - 10 * math.cos(2 * math.pi * x2) + 10)


bounds4 = [(-5.12, 5.12), (-5.12, 5.12)]
goal4 = "min"


def f5(x):
    x1, x2 = x
    # f(x1,x2) = x1*cos(x1)/20 + 2*exp(-(x1^2 + (x2-1)^2)) + 0.01*x1*x2
    return (x1 * math.cos(x1) / 20.0) + 2 * math.exp(-(x1**2 + (x2 - 1) ** 2)) + 0.01 * x1 * x2


bounds5 = [(-10, 10), (-10, 10)]
goal5 = "max"


def f6(x):
    x1, x2 = x
    # f(x1,x2) = x1*sin(4*pi*x1) - x2*sin(4*pi*x2 + pi) + 1
    return x1 * math.sin(4 * math.pi * x1) - x2 * math.sin(4 * math.pi * x2 + math.pi) + 1.0


bounds6 = [(-1, 3), (-1, 3)]
goal6 = "max"


problems = [
    (f1, bounds1, goal1),
    (f2, bounds2, goal2),
    (f3, bounds3, goal3),
    (f4, bounds4, goal4),
    (f5, bounds5, goal5),
    (f6, bounds6, goal6),
]


def clip_to_bounds(x, bounds):
    x_clipped = np.empty_like(x, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        x_clipped[i] = np.clip(x[i], lo, hi)
    return x_clipped


def better(fx_new, fx_best, goal):
    if goal == "min":
        return fx_new < fx_best
    else:
        return fx_new > fx_best


class HillClimbing:
    def __init__(self, func, bounds, goal, eps=0.1, max_it=1000, t_no_improve=50, rng=None):
        self.func = func
        self.bounds = bounds
        self.goal = goal
        self.eps = eps
        self.max_it = max_it
        self.t_no_improve = t_no_improve
        self.rng = rng or np.random.default_rng()

        self.x_opt = self.initial_point()
        self.f_opt = self.func(self.x_opt)
        self.it = 0
        self.it_since_improve = 0

        self.history_x = [self.x_opt.copy()]
        self.history_f = [self.f_opt]

    def initial_point(self):
        x = np.array(
            [self.rng.uniform(lo, hi) for (lo, hi) in self.bounds],
            dtype=float,
        )
        return clip_to_bounds(x, self.bounds)

    def neighbor(self):
        delta = self.rng.uniform(-self.eps, self.eps, size=len(self.bounds))
        cand = clip_to_bounds(self.x_opt + delta, self.bounds)
        return cand

    def better(self, f_new, f_old):
        return better(f_new, f_old, self.goal)

    def step(self):
        if self.it >= self.max_it or self.it_since_improve >= self.t_no_improve:
            return False

        cand = self.neighbor()
        f_cand = self.func(cand)

        if self.better(f_cand, self.f_opt):
            self.x_opt = cand
            self.f_opt = f_cand
            self.it_since_improve = 0
        else:
            self.it_since_improve += 1

        self.it += 1
        self.history_x.append(self.x_opt.copy())
        self.history_f.append(self.f_opt)
        return True

    def search(self):
        while self.step():
            pass
        return self.x_opt, self.f_opt


class LocalRandomSearch:
    def __init__(self, func, bounds, goal, sigma=0.1, max_it=1000, t_no_improve=50, rng=None):
        self.func = func
        self.bounds = bounds
        self.goal = goal
        self.sigma = sigma
        self.max_it = max_it
        self.t_no_improve = t_no_improve
        self.rng = rng or np.random.default_rng()

        self.x_opt = self.sample_initial()
        self.f_opt = self.func(self.x_opt)
        self.it = 0
        self.it_since_improve = 0

        self.history_x = [self.x_opt.copy()]
        self.history_f = [self.f_opt]

    def sample_initial(self):
        x = np.array([self.rng.uniform(lo, hi) for (lo, hi) in self.bounds], dtype=float)
        return clip_to_bounds(x, self.bounds)

    def perturb(self):
        step = self.rng.normal(loc=0.0, scale=self.sigma, size=len(self.bounds))
        cand = clip_to_bounds(self.x_opt + step, self.bounds)
        return cand

    def better(self, f_new, f_old):
        return better(f_new, f_old, self.goal)

    def step(self):
        if self.it >= self.max_it or self.it_since_improve >= self.t_no_improve:
            return False

        cand = self.perturb()
        f_cand = self.func(cand)

        if self.better(f_cand, self.f_opt):
            self.x_opt = cand
            self.f_opt = f_cand
            self.it_since_improve = 0
        else:
            self.it_since_improve += 1

        self.it += 1
        self.history_x.append(self.x_opt.copy())
        self.history_f.append(self.f_opt)
        return True

    def search(self):
        while self.step():
            pass
        return self.x_opt, self.f_opt


class GlobalRandomSearch:
    def __init__(self, func, bounds, goal, max_it=1000, t_no_improve=50, rng=None):
        self.func = func
        self.bounds = bounds
        self.goal = goal
        self.max_it = max_it
        self.t_no_improve = t_no_improve
        self.rng = rng or np.random.default_rng()

        self.x_opt = self.sample_point()
        self.f_opt = self.func(self.x_opt)

        self.it = 0
        self.it_since_improve = 0
        self.history_x = [self.x_opt.copy()]
        self.history_f = [self.f_opt]

    def sample_point(self):
        x = np.array([self.rng.uniform(lo, hi) for (lo, hi) in self.bounds], dtype=float)
        return clip_to_bounds(x, self.bounds)

    def better(self, f_new, f_old):
        return better(f_new, f_old, self.goal)

    def step(self):
        if self.it >= self.max_it or self.it_since_improve >= self.t_no_improve:
            return False

        cand = self.sample_point()
        f_cand = self.func(cand)

        if self.better(f_cand, self.f_opt):
            self.x_opt = cand
            self.f_opt = f_cand
            self.it_since_improve = 0
        else:
            self.it_since_improve += 1

        self.it += 1
        self.history_x.append(self.x_opt.copy())
        self.history_f.append(self.f_opt)

        return True

    def search(self):
        while self.step():
            pass
        return self.x_opt, self.f_opt


def run_experiment_problem(
    func,
    bounds,
    goal,
    R=100,
    max_it=1000,
    t_no_improve=50,
    eps_hc=0.1,
    sigma_lrs=0.1,
    sigma_grs=0.1,
    seed=42,
):
    rng = np.random.default_rng(seed)

    results = {
        "HC": [],
        "LRS": [],
        "GRS": [],
    }

    for _ in range(R):
        hc = HillClimbing(
            func=func,
            bounds=bounds,
            goal=goal,
            eps=eps_hc,
            max_it=max_it,
            t_no_improve=t_no_improve,
            rng=rng,
        )
        x_hc, fx_hc = hc.search()
        results["HC"].append((x_hc, fx_hc))

        lrs = LocalRandomSearch(
            func=func,
            bounds=bounds,
            goal=goal,
            sigma=sigma_lrs,
            max_it=max_it,
            t_no_improve=t_no_improve,
            rng=rng,
        )
        x_lrs, fx_lrs = lrs.search()
        results["LRS"].append((x_lrs, fx_lrs))

        grs = GlobalRandomSearch(
            func=func,
            bounds=bounds,
            goal=goal,
            max_it=max_it,
            t_no_improve=t_no_improve,
            rng=rng,
        )
        x_grs, fx_grs = grs.search()
        results["GRS"].append((x_grs, fx_grs))

    return results


def results_mode(results, decimals=3):
    modes = {}
    for alg_name, runs in results.items():
        keys = []
        for x, fx in runs:
            x_rounded = tuple(np.round(x, decimals))
            fx_rounded = round(float(fx), decimals)
            keys.append((x_rounded, fx_rounded))
        counts = Counter(keys)
        mode, freq = counts.most_common(1)[0]
        modes[alg_name] = (mode, freq)
    return modes


def find_best_hyperparameter_for_problem(
    problem_idx,
    func,
    bounds,
    goal,
    algorithm_type,
    hp_list,
    R=100,
    max_it=1000,
    t_no_improve=50,
    decimals=3,
    base_seed=1000,
):
    records = []

    for i_hp, hp in enumerate(hp_list):
        seed = base_seed + problem_idx * 10 + i_hp
        rng = np.random.default_rng(seed)

        results = {algorithm_type: []}

        for _ in range(R):
            if algorithm_type == "HC":
                optimizer = HillClimbing(
                    func=func,
                    bounds=bounds,
                    goal=goal,
                    eps=hp,
                    max_it=max_it,
                    t_no_improve=t_no_improve,
                    rng=rng,
                )
            elif algorithm_type == "LRS":
                optimizer = LocalRandomSearch(
                    func=func,
                    bounds=bounds,
                    goal=goal,
                    sigma=hp,
                    max_it=max_it,
                    t_no_improve=t_no_improve,
                    rng=rng,
                )
            elif algorithm_type == "GRS":
                optimizer = GlobalRandomSearch(
                    func=func,
                    bounds=bounds,
                    goal=goal,
                    max_it=max_it,
                    t_no_improve=t_no_improve,
                    rng=rng,
                )
            else:
                raise ValueError("algorithm_type deve ser 'HC', 'LRS' ou 'GRS'.")

            x, fx = optimizer.search()
            results[algorithm_type].append((x, fx))

        modes = results_mode(results, decimals=decimals)
        (x_rounded, f_rounded), freq = modes[algorithm_type]
        proportion = freq / R

        records.append(
            {
                "problem": problem_idx,
                "algorithm": algorithm_type,
                "hyperparameter": hp,
                "x_mode": x_rounded,
                "f_mode": f_rounded,
                "freq_mode": freq,
                "R": R,
                "proportion": proportion,
            }
        )

    return pd.DataFrame(records)


def select_smallest_good_hp(df_hp):
    rows = []

    for (problem, algorithm), group in df_hp.groupby(["problem", "algorithm"]):
        group_sorted = group.sort_values(by=["proportion", "hyperparameter"], ascending=[False, True])
        best = group_sorted.iloc[0]
        rows.append(best)

    return pd.DataFrame(rows).reset_index(drop=True)


def get_problem_function(problem_idx):
    func, bounds, goal = problems[problem_idx - 1]
    return func, bounds, goal


def plot_problem_with_modes(problem_idx, df_modes, grid_points=80):
    func, bounds, goal = get_problem_function(problem_idx)
    (x1_min, x1_max), (x2_min, x2_max) = bounds

    # Gera grade (x1, x2) e calcula f(x1, x2)
    x1 = np.linspace(x1_min, x1_max, grid_points)
    x2 = np.linspace(x2_min, x2_max, grid_points)
    X1, X2 = np.meshgrid(x1, x2)

    Z = np.zeros_like(X1, dtype=float)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func(np.array([X1[i, j], X2[i, j]]))

    df_p = df_modes[df_modes["indice_problema"] == problem_idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.6)

    colors = {"HC": "red", "LRS": "blue", "GRS": "green"}
    for _, row in df_p.iterrows():
        alg = row["algoritmo"]
        x_moda = row["x_moda"]  # tupla (x1, x2)
        fx_moda = row["fx_moda"]

        ax.scatter(
            x_moda[0],
            x_moda[1],
            fx_moda,
            color=colors.get(alg, "black"),
            s=60,
            label=alg,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Algoritmo")

    objetivo_str = "Minimização" if goal == "min" else "Maximização"
    ax.set_title(f"Problema {problem_idx} - {objetivo_str} - Superfície e modas")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")

    plt.tight_layout()
    plt.show()


def main():
    eps_list_hc = [0.5, 0.2, 0.1, 0.05, 0.01]
    sigma_list_lrs = [0.5, 0.2, 0.1, 0.05, 0.01]
    sigma_list_grs = [0.5, 0.2, 0.1, 0.05, 0.01]

    R = 100
    max_it = 1000
    t_no_improve = 50

    dfs_hp = []

    for idx, (func, bounds, goal) in enumerate(problems, start=1):
        df_hc = find_best_hyperparameter_for_problem(
            problem_idx=idx,
            func=func,
            bounds=bounds,
            goal=goal,
            algorithm_type="HC",
            hp_list=eps_list_hc,
            R=R,
            max_it=max_it,
            t_no_improve=t_no_improve,
            decimals=3,
            base_seed=2000,
        )
        dfs_hp.append(df_hc)

        df_lrs = find_best_hyperparameter_for_problem(
            problem_idx=idx,
            func=func,
            bounds=bounds,
            goal=goal,
            algorithm_type="LRS",
            hp_list=sigma_list_lrs,
            R=R,
            max_it=max_it,
            t_no_improve=t_no_improve,
            decimals=3,
            base_seed=3000,
        )
        dfs_hp.append(df_lrs)

        df_grs = find_best_hyperparameter_for_problem(
            problem_idx=idx,
            func=func,
            bounds=bounds,
            goal=goal,
            algorithm_type="GRS",
            hp_list=sigma_list_grs,
            R=R,
            max_it=max_it,
            t_no_improve=t_no_improve,
            decimals=3,
            base_seed=4000,
        )
        dfs_hp.append(df_grs)

    df_hyperparameters = pd.concat(dfs_hp, ignore_index=True)
    df_best_hps = select_smallest_good_hp(df_hyperparameters)

    modes_table = []

    for idx, (func, bounds, goal) in enumerate(problems, start=1):
        best_hp_problem = df_best_hps[df_best_hps["problem"] == idx]

        eps_hc = best_hp_problem.loc[best_hp_problem["algorithm"] == "HC", "hyperparameter"].iloc[0]
        sigma_lrs = best_hp_problem.loc[best_hp_problem["algorithm"] == "LRS", "hyperparameter"].iloc[0]
        sigma_grs = best_hp_problem.loc[best_hp_problem["algorithm"] == "GRS", "hyperparameter"].iloc[0]

        results = run_experiment_problem(
            func,
            bounds,
            goal,
            R=R,
            max_it=max_it,
            t_no_improve=t_no_improve,
            eps_hc=eps_hc,
            sigma_lrs=sigma_lrs,
            sigma_grs=sigma_grs,
            seed=42 + idx,
        )

        modes = results_mode(results, decimals=3)

        for alg, (sol, freq) in modes.items():
            x_rounded, fx_rounded = sol

            if alg == "HC":
                hp_value = eps_hc
                hp_name = "eps"
            elif alg == "LRS":
                hp_value = sigma_lrs
                hp_name = "sigma"
            else:
                hp_value = sigma_grs
                hp_name = "sigma"

            modes_table.append(
                {
                    "indice_problema": idx,
                    "objetivo": goal,
                    "algoritmo": alg,
                    "nome_hiperparametro": hp_name,
                    "valor_hiperparametro": hp_value,
                    "x_moda": x_rounded,
                    "fx_moda": fx_rounded,
                    "frequencia_moda": freq,
                    "numero_rodadas": R,
                }
            )

    df_modes = pd.DataFrame(modes_table)

    print("Melhores hiperparâmetros por problema/algoritmo:")
    print(df_best_hps)
    print("\nTabela de modas:")
    print(df_modes)

    for i in range(1, 7):
        plot_problem_with_modes(i, df_modes)


if __name__ == "__main__":
    main()
