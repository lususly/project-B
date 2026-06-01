import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    n_values: List[int]
    topic_num: int = 3

    # Adaptive repeated experiments
    min_repeats: int = 10
    max_repeats: int = 30
    repeat_batch: int = 5
    relative_ci_target: float = 0.10
    absolute_ci_target: float = 0.002

    # Adaptive convergence detection
    seed: int = 42
    max_iter: int = 500
    min_iter: int = 10
    abs_tol: float = 1e-8
    rel_tol: float = 1e-7
    patience: int = 5

    # Slope fitting
    fit_start_ratio: float = 0.2
    fit_end_ratio: float = 0.9

    # Output control
    save_full_trajectories: bool = False
    save_single_run_figures: bool = False
    save_mean_figures: bool = True
    save_network_schematics: bool = True
    realtime_save_every: int = 20


# ============================================================
# 2. Output Directory
# ============================================================

def get_script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def prepare_output_dirs() -> Dict[str, str]:
    base_dir = os.path.join(get_script_dir(), "degroot_outputs_ver11")

    paths = {
        "base": base_dir,
        "single_figures": os.path.join(base_dir, "figures_single_run"),
        "mean_figures": os.path.join(base_dir, "figures_mean_results"),
        "summary_data": os.path.join(base_dir, "summary_data"),
        "analysis_results": os.path.join(base_dir, "analysis_results"),
        "raw_trajectories": os.path.join(base_dir, "raw_trajectories"),
        "matrices": os.path.join(base_dir, "matrices"),
        "network_schematics": os.path.join(base_dir, "network_schematics")
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


# ============================================================
# 3. Basic Utilities
# ============================================================

def row_normalize(W: np.ndarray) -> np.ndarray:
    W = np.array(W, dtype=float)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def normalize_logic_matrix(C: np.ndarray) -> np.ndarray:
    """
    Enforce the logic-matrix condition used in the multidimensional
    DeGroot literature: each row has absolute sum 1 and every diagonal
    entry remains positive.
    """
    C = np.array(C, dtype=float)

    for i in range(C.shape[0]):
        row_abs_sum = np.sum(np.abs(C[i]))

        if row_abs_sum == 0:
            C[i, i] = 1.0
            row_abs_sum = 1.0

        C[i] = C[i] / row_abs_sum

        if C[i, i] <= 0:
            C[i, i] = 1e-6
            C[i] = C[i] / np.sum(np.abs(C[i]))

    return C


def check_strong_connectivity(W: np.ndarray) -> bool:
    G = nx.from_numpy_array(W > 0, create_using=nx.DiGraph)
    return nx.is_strongly_connected(G)


def matrix_euclidean_norm(M: np.ndarray) -> float:
    """Euclidean norm of a matrix after vectorisation."""
    return float(np.linalg.norm(np.asarray(M).reshape(-1), ord=2))


# ============================================================
# 4. Network Generation
# ============================================================

def add_directed_cycle(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    for i in range(n):
        A[i, (i + 1) % n] = 1
    return A


def add_all_self_loops(A: np.ndarray) -> np.ndarray:
    """
    Add positive self-confidence links for all agents, matching the
    standard assumption w_ii > 0 for every individual.
    """
    np.fill_diagonal(A, 1)
    return A


def generate_cycle_baseline(n: int, rng: np.random.Generator) -> np.ndarray:
    A = np.zeros((n, n))
    A = add_directed_cycle(A)
    A = add_all_self_loops(A)

    weights = rng.uniform(0.2, 1.0, size=(n, n))
    W = row_normalize(A * weights)
    return W


def generate_random_strong(
    n: int,
    rng: np.random.Generator,
    extra_edge_prob: float = 0.08
) -> np.ndarray:
    A = np.zeros((n, n))
    A = add_directed_cycle(A)

    random_edges = rng.random((n, n)) < extra_edge_prob
    np.fill_diagonal(random_edges, False)
    A[random_edges] = 1

    A = add_all_self_loops(A)

    weights = rng.uniform(0.1, 1.0, size=(n, n))
    W = row_normalize(A * weights)
    return W


def generate_small_world(
    n: int,
    rng: np.random.Generator,
    k: int = 4,
    rewire_prob: float = 0.15
) -> np.ndarray:
    A = np.zeros((n, n))
    A = add_directed_cycle(A)

    k = min(k, max(1, n - 1))

    for i in range(n):
        for d in range(1, k + 1):
            A[i, (i + d) % n] = 1

    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] == 1 and rng.random() < rewire_prob:
                A[i, j] = 0

                new_j = rng.integers(0, n)
                while new_j == i:
                    new_j = rng.integers(0, n)

                A[i, new_j] = 1

    # Re-add directed cycle to guarantee strong connectivity.
    A = add_directed_cycle(A)
    A = add_all_self_loops(A)

    weights = rng.uniform(0.1, 1.0, size=(n, n))
    W = row_normalize(A * weights)
    return W


def generate_scale_free(
    n: int,
    rng: np.random.Generator,
    m_attach: int = 2
) -> np.ndarray:
    A = np.zeros((n, n))
    A = add_directed_cycle(A)

    if n <= 2:
        A = add_all_self_loops(A)
        weights = rng.uniform(0.1, 1.0, size=(n, n))
        return row_normalize(A * weights)

    degrees = np.ones(n)

    for new_node in range(1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        attach_count = min(m_attach, new_node)

        targets = rng.choice(
            np.arange(new_node),
            size=attach_count,
            replace=False,
            p=probs
        )

        for target in targets:
            A[new_node, target] = 1

            if rng.random() < 0.5:
                A[target, new_node] = 1

            degrees[target] += 1
            degrees[new_node] += 1

    # Re-add directed cycle to guarantee strong connectivity.
    A = add_directed_cycle(A)
    A = add_all_self_loops(A)

    weights = rng.uniform(0.1, 1.0, size=(n, n))
    W = row_normalize(A * weights)
    return W


def generate_network(
    n: int,
    network_type: str,
    rng: np.random.Generator
) -> np.ndarray:
    if network_type == "cycle_baseline":
        W = generate_cycle_baseline(n, rng)
    elif network_type == "random_strong":
        W = generate_random_strong(n, rng)
    elif network_type == "small_world":
        W = generate_small_world(n, rng)
    elif network_type == "scale_free":
        W = generate_scale_free(n, rng)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    if not check_strong_connectivity(W):
        raise RuntimeError(f"{network_type} network with n={n} is not strongly connected.")

    if not np.allclose(W.sum(axis=1), 1.0):
        raise RuntimeError("W is not row-stochastic.")

    if not np.all(np.diag(W) > 0):
        raise RuntimeError("The literature condition w_ii > 0 for every agent is not satisfied.")

    return W


# ============================================================
# 5. Logic Matrix Construction
# ============================================================

def make_logic_matrices(
    n: int,
    logic_type: str,
    rng: np.random.Generator,
    a: float = 0.6,
    b: float = 0.4,
    heterogeneity: float = 0.08
) -> np.ndarray:
    """
    Construct three logic-matrix settings:
    1. independent: all topics are independent;
    2. cascade: all agents share the same cascade dependency;
    3. heterogeneous_cascade: agents share the cascade sparsity pattern
       but have random individual weights.
    """
    if logic_type == "independent":
        C = np.eye(3)
        C_all = np.repeat(C[None, :, :], n, axis=0)

    elif logic_type == "cascade":
        C = np.array([
            [1.0, 0.0, 0.0],
            [a, 1.0 - a, 0.0],
            [b, 0.0, 1.0 - b]
        ])
        C = normalize_logic_matrix(C)
        C_all = np.repeat(C[None, :, :], n, axis=0)

    elif logic_type == "heterogeneous_cascade":
        base_C = np.array([
            [1.0, 0.0, 0.0],
            [a, 1.0 - a, 0.0],
            [b, 0.0, 1.0 - b]
        ])

        C_all = []
        mask = (base_C != 0).astype(float)

        for _ in range(n):
            noise = rng.normal(0.0, heterogeneity, size=(3, 3))
            C_i = base_C + noise * mask
            C_i = normalize_logic_matrix(C_i)
            C_all.append(C_i)

        C_all = np.array(C_all)

    else:
        raise ValueError(f"Unknown logic type: {logic_type}")

    return C_all


# ============================================================
# 6. Multidimensional DeGroot Simulation
# ============================================================

def multidim_update(
    X: np.ndarray,
    W: np.ndarray,
    C_all: np.ndarray
) -> np.ndarray:
    """
    Literature-aligned update:
        x_i(t+1) = sum_j w_ij C_i x_j(t)

    This is algebraically equivalent to applying C_i after the social
    average because C_i is fixed with respect to j, but this explicit
    implementation matches the formula in the literature.
    """
    n, m = X.shape
    X_next = np.zeros_like(X)

    for i in range(n):
        updated_opinion = np.zeros(m)
        for j in range(n):
            updated_opinion += W[i, j] * (C_all[i] @ X[j, :])
        X_next[i, :] = updated_opinion

    return X_next


def run_until_convergence(
    W: np.ndarray,
    C_all: np.ndarray,
    X0: np.ndarray,
    abs_tol: float,
    rel_tol: float,
    patience: int,
    min_iter: int,
    max_iter: int
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    X = X0.copy()
    trajectory = [X.copy()]
    distances = []

    stable_count = 0
    converged = False
    convergence_time = max_iter

    for t in range(max_iter):
        X_next = multidim_update(X, W, C_all)

        # Euclidean norm after vectorising the opinion-state matrix.
        dist = matrix_euclidean_norm(X_next - X)
        scale = max(matrix_euclidean_norm(X), 1.0)
        threshold = abs_tol + rel_tol * scale

        distances.append(dist)
        trajectory.append(X_next.copy())

        if t + 1 >= min_iter and dist < threshold:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= patience:
            converged = True
            convergence_time = t + 1
            break

        X = X_next

    return np.array(trajectory), np.array(distances), convergence_time, converged


# ============================================================
# 7. Metrics
# ============================================================

def compute_final_disagreement(X_final: np.ndarray) -> float:
    """
    Mean Euclidean distance from each final opinion vector to the final
    population mean. This measures residual inter-agent disagreement,
    not the difference between final and initial opinions.
    """
    mean_opinion = X_final.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(X_final - mean_opinion, axis=1)))


def compute_topic_disagreement(X_final: np.ndarray) -> np.ndarray:
    """
    Topic-wise standard deviation across agents at the final time.
    This measures which topics retain stronger inter-agent disagreement.
    """
    return np.std(X_final, axis=0)


def fit_log_slope(
    distances: np.ndarray,
    fit_start_ratio: float,
    fit_end_ratio: float,
    eps: float = 1e-12
) -> Tuple[float, float]:
    if len(distances) < 5:
        return np.nan, np.nan

    y = np.log(distances + eps)
    t = np.arange(len(y))

    start = int(len(y) * fit_start_ratio)
    end = int(len(y) * fit_end_ratio)

    if end <= start + 2:
        start = 0
        end = len(y)

    x_fit = t[start:end]
    y_fit = y[start:end]

    valid = np.isfinite(y_fit)

    if valid.sum() < 3:
        return np.nan, np.nan

    slope, intercept = np.polyfit(x_fit[valid], y_fit[valid], 1)
    return float(slope), float(intercept)


def summarize_single_run_v11(
    trajectory: np.ndarray,
    distances: np.ndarray,
    convergence_time: int,
    converged: bool,
    config: ExperimentConfig
) -> Dict:
    X_final = trajectory[-1]

    slope, intercept = fit_log_slope(
        distances,
        config.fit_start_ratio,
        config.fit_end_ratio
    )

    topic_dis = compute_topic_disagreement(X_final)

    return {
        "convergence_time": convergence_time,
        "converged": converged,
        "slope": slope,
        "intercept": intercept,
        "final_disagreement": compute_final_disagreement(X_final),
        "topic1_disagreement": topic_dis[0],
        "topic2_disagreement": topic_dis[1],
        "topic3_disagreement": topic_dis[2],
        "final_distance": float(distances[-1]) if len(distances) > 0 else np.nan,
        "iterations_recorded": len(distances)
    }


def enough_repeats_by_ci(
    slopes: List[float],
    min_repeats: int,
    relative_ci_target: float,
    absolute_ci_target: float
) -> Tuple[bool, float, float]:
    valid_slopes = np.array([s for s in slopes if np.isfinite(s)])

    if len(valid_slopes) < min_repeats:
        return False, np.nan, np.nan

    mean_slope = np.mean(valid_slopes)
    std_slope = np.std(valid_slopes, ddof=1)
    ci_half_width = 1.96 * std_slope / np.sqrt(len(valid_slopes))

    target = max(
        absolute_ci_target,
        relative_ci_target * max(abs(mean_slope), 1e-12)
    )

    enough = ci_half_width <= target
    return enough, float(ci_half_width), float(target)


# ============================================================
# 8. Variable-Length Averaging
# ============================================================

def average_variable_length_trajectories(
    trajectories: List[np.ndarray]
) -> np.ndarray:
    """
    Each element shape: (T_i, 3), already averaged over agents.
    Return shape: (max_T, 3).
    Shorter trajectories are padded by their final stable value.
    """
    max_len = max(traj.shape[0] for traj in trajectories)
    padded = []

    for traj in trajectories:
        current_len = traj.shape[0]

        if current_len < max_len:
            pad = np.repeat(traj[-1:, :], max_len - current_len, axis=0)
            traj_pad = np.vstack([traj, pad])
        else:
            traj_pad = traj

        padded.append(traj_pad)

    return np.mean(np.array(padded), axis=0)


def average_variable_length_distances(
    distances_list: List[np.ndarray]
) -> np.ndarray:
    """
    Average distance curves with variable lengths.
    Shorter runs are padded by their final distance.
    """
    max_len = max(len(d) for d in distances_list)
    padded = []

    for d in distances_list:
        if len(d) < max_len:
            pad_value = d[-1] if len(d) > 0 else np.nan
            d_pad = np.concatenate([d, np.repeat(pad_value, max_len - len(d))])
        else:
            d_pad = d

        padded.append(d_pad)

    return np.nanmean(np.array(padded), axis=0)


# ============================================================
# 9. Plotting
# ============================================================

def add_bar_value_labels(ax, fmt: str = "{:.4f}") -> None:
    """Add numeric labels to every bar in a bar chart."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0

    for bar in ax.patches:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        offset = 0.02 * span
        va = "bottom" if height >= 0 else "top"
        y = height + offset if height >= 0 else height - offset
        ax.text(x, y, fmt.format(height), ha="center", va=va, fontsize=8, rotation=0)


def plot_single_topic_trajectories(
    trajectory: np.ndarray,
    save_path: str,
    title: str
) -> None:
    T, n, m = trajectory.shape
    t = np.arange(T)

    plt.figure(figsize=(10, 6))

    for topic in range(m):
        values = trajectory[:, :, topic]
        mean_values = values.mean(axis=1)

        if n <= 25:
            for i in range(n):
                plt.plot(t, values[:, i], alpha=0.15, linewidth=0.6)

        plt.plot(t, mean_values, linewidth=2.5, label=f"Topic {topic + 1} mean")

    plt.xlabel("Time step")
    plt.ylabel("Opinion value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_single_log_distance(
    distances: np.ndarray,
    slope: float,
    intercept: float,
    save_path: str,
    title: str
) -> None:
    y = np.log(distances + 1e-12)
    t = np.arange(len(y))

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label="log Euclidean distance")

    if np.isfinite(slope):
        plt.plot(
            t,
            intercept + slope * t,
            linestyle="--",
            label=f"fit slope = {slope:.5f}"
        )

    plt.xlabel("Time step")
    plt.ylabel("log(||vec(X(t+1)-X(t))||_2)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mean_topic_trajectories(
    mean_topic_trajectory: np.ndarray,
    save_path: str,
    title: str
) -> None:
    t = np.arange(mean_topic_trajectory.shape[0])

    plt.figure(figsize=(10, 6))

    for topic in range(mean_topic_trajectory.shape[1]):
        plt.plot(
            t,
            mean_topic_trajectory[:, topic],
            linewidth=2.5,
            label=f"Topic {topic + 1} mean"
        )

    plt.xlabel("Time step")
    plt.ylabel("Mean opinion value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mean_log_distance(
    mean_distances: np.ndarray,
    save_path: str,
    title: str,
    config: ExperimentConfig
) -> Tuple[float, float]:
    slope, intercept = fit_log_slope(
        mean_distances,
        config.fit_start_ratio,
        config.fit_end_ratio
    )

    y = np.log(mean_distances + 1e-12)
    t = np.arange(len(y))

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label="mean log Euclidean distance")

    if np.isfinite(slope):
        plt.plot(
            t,
            intercept + slope * t,
            linestyle="--",
            label=f"mean fit slope = {slope:.5f}"
        )

    plt.xlabel("Time step")
    plt.ylabel("log(mean ||vec(X(t+1)-X(t))||_2)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return slope, intercept


def plot_network_schematic(W: np.ndarray, save_path: str, title: str) -> None:
    """Draw a simple directed schematic for a generated network."""
    G = nx.from_numpy_array(W > 0, create_using=nx.DiGraph)
    n = W.shape[0]

    # Draw non-self-loop edges for readability; self-loops are indicated in the note.
    self_loop_nodes = [i for i in range(n) if W[i, i] > 0]
    G_no_loops = G.copy()
    G_no_loops.remove_edges_from(nx.selfloop_edges(G_no_loops))

    pos = nx.spring_layout(G_no_loops, seed=42)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G_no_loops, pos, node_size=600)
    nx.draw_networkx_labels(G_no_loops, pos, font_size=9)
    nx.draw_networkx_edges(
        G_no_loops,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=12,
        width=1.2,
        alpha=0.75,
        connectionstyle="arc3,rad=0.08"
    )
    plt.title(title)
    plt.text(
        0.5,
        -0.08,
        f"n={n}; every node has positive self-loop: {len(self_loop_nodes)}/{n}",
        transform=plt.gca().transAxes,
        ha="center",
        fontsize=9
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_network_schematic_figures(paths: Dict[str, str], n: int = 12, seed: int = 42) -> None:
    """
    Generate three representative network schematics used in the experiment:
    random strong, small-world, and scale-free. The cycle baseline is a
    control benchmark and is not included here unless separately requested.
    """
    rng = np.random.default_rng(seed)
    network_types = ["random_strong", "small_world", "scale_free"]

    for network_type in network_types:
        W = generate_network(n, network_type, rng)
        readable_title = network_type.replace("_", " ").title()
        plot_network_schematic(
            W,
            os.path.join(paths["network_schematics"], f"schematic_{network_type}_n{n}.png"),
            f"Representative {readable_title} Network"
        )


def plot_factor_effects(grouped_df: pd.DataFrame, paths: Dict[str, str]) -> None:
    # Agent number effect
    n_effect = grouped_df.groupby("n", as_index=False).agg(
        mean_slope=("mean_slope", "mean"),
        mean_convergence_time=("mean_convergence_time", "mean"),
        mean_final_disagreement=("mean_final_disagreement", "mean"),
        mean_actual_repeats=("actual_repeats", "mean")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(n_effect["n"], n_effect["mean_slope"], marker="o")
    plt.xlabel("Number of agents")
    plt.ylabel("Mean convergence slope")
    plt.title("Effect of Agent Number on Convergence Speed")
    plt.tight_layout()
    plt.savefig(
        os.path.join(paths["analysis_results"], "effect_agent_number_on_slope.png"),
        dpi=300
    )
    plt.close()

    # Network type effect
    network_effect = grouped_df.groupby("network_type", as_index=False).agg(
        mean_slope=("mean_slope", "mean"),
        mean_convergence_time=("mean_convergence_time", "mean"),
        mean_final_disagreement=("mean_final_disagreement", "mean"),
        mean_actual_repeats=("actual_repeats", "mean")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(network_effect["network_type"], network_effect["mean_slope"])
    ax.set_xlabel("Network type")
    ax.set_ylabel("Mean convergence slope")
    ax.set_title("Effect of Network Type on Convergence Speed")
    ax.tick_params(axis="x", rotation=30)
    add_bar_value_labels(ax, fmt="{:.5f}")
    fig.tight_layout()
    fig.savefig(
        os.path.join(paths["analysis_results"], "effect_network_type_on_slope.png"),
        dpi=300
    )
    plt.close(fig)

    # Logic matrix effect
    logic_effect = grouped_df.groupby("logic_type", as_index=False).agg(
        mean_slope=("mean_slope", "mean"),
        mean_convergence_time=("mean_convergence_time", "mean"),
        mean_final_disagreement=("mean_final_disagreement", "mean"),
        mean_actual_repeats=("actual_repeats", "mean")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(logic_effect["logic_type"], logic_effect["mean_slope"])
    ax.set_xlabel("Logic matrix type")
    ax.set_ylabel("Mean convergence slope")
    ax.set_title("Effect of Logic Matrix Type on Convergence Speed")
    ax.tick_params(axis="x", rotation=30)
    add_bar_value_labels(ax, fmt="{:.5f}")
    fig.tight_layout()
    fig.savefig(
        os.path.join(paths["analysis_results"], "effect_logic_type_on_slope.png"),
        dpi=300
    )
    plt.close(fig)

    # Network and logic interaction heatmap
    pivot = grouped_df.pivot_table(
        index="network_type",
        columns="logic_type",
        values="mean_slope",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean convergence slope")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Interaction Effect: Network Type and Logic Matrix Type")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.4f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        os.path.join(paths["analysis_results"], "interaction_network_logic_heatmap.png"),
        dpi=300
    )
    plt.close(fig)

def plot_controlled_slope_effects(grouped_df: pd.DataFrame, paths: Dict[str, str]) -> None:

    network_order = [
        "cycle_baseline",
        "random_strong",
        "small_world",
        "scale_free"
    ]

    logic_order = [
        "independent",
        "cascade",
        "heterogeneous_cascade"
    ]

    # 1. Same logic type, compare four network structures as n increases
    for logic_type in logic_order:
        subset = grouped_df[grouped_df["logic_type"] == logic_type]

        plt.figure(figsize=(10, 6))

        for network_type in network_order:
            data = subset[subset["network_type"] == network_type].sort_values("n")

            plt.plot(
                data["n"],
                data["mean_slope"],
                marker="o",
                linewidth=2,
                label=network_type
            )

        plt.xlabel("Number of agents")
        plt.ylabel("Mean convergence slope")
        plt.title(f"Network Effect on Convergence Speed | Logic = {logic_type}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(paths["analysis_results"],
                         f"controlled_network_effect_logic_{logic_type}.png"),
            dpi=300
        )
        plt.close()

    # 2. Same network type, compare three logic structures as n increases
    for network_type in network_order:
        subset = grouped_df[grouped_df["network_type"] == network_type]

        plt.figure(figsize=(10, 6))

        for logic_type in logic_order:
            data = subset[subset["logic_type"] == logic_type].sort_values("n")

            plt.plot(
                data["n"],
                data["mean_slope"],
                marker="o",
                linewidth=2,
                label=logic_type
            )

        plt.xlabel("Number of agents")
        plt.ylabel("Mean convergence slope")
        plt.title(f"Logic Matrix Effect on Convergence Speed | Network = {network_type}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(paths["analysis_results"],
                         f"controlled_logic_effect_network_{network_type}.png"),
            dpi=300
        )
        plt.close()


# ============================================================
# 10. Main Experiment Runner
# ============================================================

def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    paths = prepare_output_dirs()

    if config.save_network_schematics:
        generate_network_schematic_figures(paths, n=12, seed=config.seed)

    network_types = [
        "cycle_baseline",
        "random_strong",
        "small_world",
        "scale_free"
    ]

    logic_types = [
        "independent",
        "cascade",
        "heterogeneous_cascade"
    ]

    all_results = []
    mean_results = []

    master_rng = np.random.default_rng(config.seed)
    completed_runs = 0

    for n in config.n_values:
        for network_type in network_types:
            for logic_type in logic_types:
                group_topic_trajectories = []
                group_distances = []
                group_slopes = []

                repeat = 0
                stop_reason = "max_repeats_reached"

                while repeat < config.max_repeats:
                    run_seed = int(master_rng.integers(0, 1_000_000_000))
                    rng = np.random.default_rng(run_seed)

                    W = generate_network(n, network_type, rng)
                    C_all = make_logic_matrices(n, logic_type, rng)
                    X0 = rng.uniform(-1.0, 1.0, size=(n, config.topic_num))

                    trajectory, distances, convergence_time, converged = run_until_convergence(
                        W=W,
                        C_all=C_all,
                        X0=X0,
                        abs_tol=config.abs_tol,
                        rel_tol=config.rel_tol,
                        patience=config.patience,
                        min_iter=config.min_iter,
                        max_iter=config.max_iter
                    )

                    result = summarize_single_run_v11(
                        trajectory=trajectory,
                        distances=distances,
                        convergence_time=convergence_time,
                        converged=converged,
                        config=config
                    )

                    result.update({
                        "n": n,
                        "network_type": network_type,
                        "logic_type": logic_type,
                        "repeat": repeat,
                        "seed": run_seed
                    })

                    all_results.append(result)
                    group_slopes.append(result["slope"])
                    group_topic_trajectories.append(trajectory.mean(axis=1))
                    group_distances.append(distances)

                    run_id = f"n{n}_{network_type}_{logic_type}_r{repeat}"

                    if repeat == 0:
                        np.save(
                            os.path.join(paths["matrices"], f"{run_id}_W.npy"),
                            W
                        )
                        np.save(
                            os.path.join(paths["matrices"], f"{run_id}_C_all.npy"),
                            C_all
                        )

                    if config.save_single_run_figures and repeat == 0:
                        plot_single_topic_trajectories(
                            trajectory,
                            os.path.join(paths["single_figures"], f"{run_id}_topic_trajectories.png"),
                            f"Single Run Topic Trajectories | n={n}, {network_type}, {logic_type}"
                        )

                        plot_single_log_distance(
                            distances,
                            result["slope"],
                            result["intercept"],
                            os.path.join(paths["single_figures"], f"{run_id}_log_distance_fit.png"),
                            f"Single Run Log-Distance Fit | n={n}, {network_type}, {logic_type}"
                        )

                    if config.save_full_trajectories:
                        np.save(
                            os.path.join(paths["raw_trajectories"], f"{run_id}_trajectory.npy"),
                            trajectory
                        )
                        np.save(
                            os.path.join(paths["raw_trajectories"], f"{run_id}_distances.npy"),
                            distances
                        )

                    repeat += 1
                    completed_runs += 1

                    if (
                        repeat >= config.min_repeats
                        and repeat % config.repeat_batch == 0
                    ):
                        enough, ci_half_width, ci_target = enough_repeats_by_ci(
                            group_slopes,
                            config.min_repeats,
                            config.relative_ci_target,
                            config.absolute_ci_target
                        )

                        if enough:
                            stop_reason = "ci_stable"
                            break

                    if completed_runs % config.realtime_save_every == 0:
                        pd.DataFrame(all_results).to_csv(
                            os.path.join(paths["summary_data"], "summary_raw_realtime.csv"),
                            index=False
                        )

                    print(
                        f"Done | n={n:2d} | network={network_type:15s} | "
                        f"C={logic_type:22s} | repeat={repeat:2d} | "
                        f"slope={result['slope']:.5f} | "
                        f"time={convergence_time:4d} | converged={converged}"
                    )

                mean_topic_trajectory = average_variable_length_trajectories(
                    group_topic_trajectories
                )

                mean_distances = average_variable_length_distances(
                    group_distances
                )

                mean_id = f"n{n}_{network_type}_{logic_type}_mean"

                if config.save_mean_figures:
                    plot_mean_topic_trajectories(
                        mean_topic_trajectory,
                        os.path.join(paths["mean_figures"], f"{mean_id}_mean_topic_trajectories.png"),
                        f"Mean Topic Trajectories | n={n}, {network_type}, {logic_type}"
                    )

                    mean_slope_from_distance, mean_intercept_from_distance = plot_mean_log_distance(
                        mean_distances,
                        os.path.join(paths["mean_figures"], f"{mean_id}_mean_log_distance_fit.png"),
                        f"Mean Log-Distance Fit | n={n}, {network_type}, {logic_type}",
                        config
                    )
                else:
                    mean_slope_from_distance, mean_intercept_from_distance = np.nan, np.nan

                enough, ci_half_width, ci_target = enough_repeats_by_ci(
                    group_slopes,
                    config.min_repeats,
                    config.relative_ci_target,
                    config.absolute_ci_target
                )

                mean_results.append({
                    "n": n,
                    "network_type": network_type,
                    "logic_type": logic_type,
                    "actual_repeats": repeat,
                    "repeat_stop_reason": stop_reason,
                    "slope_ci_half_width": ci_half_width,
                    "slope_ci_target": ci_target,
                    "mean_slope_from_mean_distance": mean_slope_from_distance,
                    "mean_intercept_from_mean_distance": mean_intercept_from_distance
                })

    raw_df = pd.DataFrame(all_results)

    raw_df.to_csv(
        os.path.join(paths["summary_data"], "summary_raw.csv"),
        index=False
    )

    grouped_df = raw_df.groupby(
        ["n", "network_type", "logic_type"],
        as_index=False
    ).agg(
        mean_slope=("slope", "mean"),
        std_slope=("slope", "std"),
        mean_convergence_time=("convergence_time", "mean"),
        std_convergence_time=("convergence_time", "std"),
        mean_final_disagreement=("final_disagreement", "mean"),
        std_final_disagreement=("final_disagreement", "std"),
        mean_topic1_disagreement=("topic1_disagreement", "mean"),
        mean_topic2_disagreement=("topic2_disagreement", "mean"),
        mean_topic3_disagreement=("topic3_disagreement", "mean"),
        convergence_rate=("converged", "mean")
    )

    mean_distance_df = pd.DataFrame(mean_results)

    final_summary = grouped_df.merge(
        mean_distance_df,
        on=["n", "network_type", "logic_type"],
        how="left"
    )

    final_summary.to_csv(
        os.path.join(paths["summary_data"], "summary_grouped.csv"),
        index=False
    )

    network_effect = final_summary.groupby("network_type", as_index=False).agg(
        mean_slope=("mean_slope", "mean"),
        mean_convergence_time=("mean_convergence_time", "mean"),
        mean_final_disagreement=("mean_final_disagreement", "mean"),
        mean_actual_repeats=("actual_repeats", "mean")
    )

    logic_effect = final_summary.groupby("logic_type", as_index=False).agg(
        mean_slope=("mean_slope", "mean"),
        mean_convergence_time=("mean_convergence_time", "mean"),
        mean_final_disagreement=("mean_final_disagreement", "mean"),
        mean_actual_repeats=("actual_repeats", "mean")
    )

    n_effect = final_summary.groupby("n", as_index=False).agg(
        mean_slope=("mean_slope", "mean"),
        mean_convergence_time=("mean_convergence_time", "mean"),
        mean_final_disagreement=("mean_final_disagreement", "mean"),
        mean_actual_repeats=("actual_repeats", "mean")
    )

    network_effect.to_csv(
        os.path.join(paths["analysis_results"], "analysis_by_network_type.csv"),
        index=False
    )

    logic_effect.to_csv(
        os.path.join(paths["analysis_results"], "analysis_by_logic_type.csv"),
        index=False
    )

    n_effect.to_csv(
        os.path.join(paths["analysis_results"], "analysis_by_agent_number.csv"),
        index=False
    )

    plot_factor_effects(final_summary, paths)
    plot_controlled_slope_effects(final_summary, paths)

    print("\nExperiment finished.")
    print(f"All outputs are saved under:\n{paths['base']}")

    return final_summary


# ============================================================
# 11. Run
# ============================================================

if __name__ == "__main__":
    config = ExperimentConfig(
        n_values=list(range(2, 21)),
        topic_num=3,

        min_repeats=10,
        max_repeats=30,
        repeat_batch=5,
        relative_ci_target=0.10,
        absolute_ci_target=0.002,

        seed=42,
        max_iter=500,
        min_iter=10,
        abs_tol=1e-8,
        rel_tol=1e-7,
        patience=5,

        fit_start_ratio=0.2,
        fit_end_ratio=0.9,

        save_full_trajectories=False,
        save_single_run_figures=False,
        save_mean_figures=True,
        save_network_schematics=True,
        realtime_save_every=20
    )

    final_summary = run_experiment(config)
    print(final_summary.head())