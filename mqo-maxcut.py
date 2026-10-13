import torch
import networkx as nx
import numpy as np
import psutil, os
import time
from typing import Tuple, Dict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ============================================================
# Multiple ER settings (run all)
# ============================================================
SETTINGS = [
    # (30000, 0.5),
    # (30000, 0.8),
    # (40000, 0.5),
    # (40000, 0.8),
    (1000, 0.8),
    (1000, 0.5),
    (1000, 0.1),
    (100,  0.5),
]

SEEDS = list(range(8))  # [0,1,2,3,4,5,6,7]

alpha = 0.0025
beta  = 0.8
ITERATION_T = 3000
# epoch = 100

RESET_ROUNDS = 90
RESET_FRAC = 0.8
EPS_BOUNDARY = 0.4 
TIME_LIMIT = 10 * 60  # per seed

# linear bias epsilon for tie-breaking (0 < eps << 1)
EPS_LINEAR_BIAS = 1e-8

# ==========================================================================================
# Objective / Gradient
# Use adjacency + linear bias for gradient:
# f_AL(x) = x^T A x + eps * 1^T x,   x in [-1,1]^n
# f'_AL(x)= 2Ax + eps * 1
# ==========================================================================================

def loss_AL(x: torch.Tensor, A: torch.Tensor, eps: float, ones: torch.Tensor) -> torch.Tensor:
    # f_AL(x) = x^T A x + eps * 1^T x
    return (x @ (A @ x)) + eps * (ones @ x)

def grad_AL(x: torch.Tensor, A: torch.Tensor, eps: float, ones: torch.Tensor) -> torch.Tensor:
    # f'_AL(x) = 2Ax + eps * 1
    return 2.0 * (A @ x) + eps * ones

def cut_value_from_binary(x_bin: torch.Tensor, L: torch.Tensor) -> float:
    # cut(x_bin) = 1/4 * x_bin^T L x_bin
    return float(0.25 * (x_bin @ (L @ x_bin)).item())

# Check whether x is close to the binary set {-1, 1}^n:
# If the distance from each component to its nearest value (+1 or -1)
# is no greater than eps, we consider it to have converged to the binary boundary.
def is_binary_converged(x: torch.Tensor, eps: float = 1e-3) -> bool:
    dist = torch.minimum(torch.abs(x - 1.0), torch.abs(x + 1.0))
    return bool(torch.all(dist <= eps).item())


# This function measures the proportion of variables that are close to the binary boundary before full convergence.
# It is only used for hyperparameter tuning and can be commented out afterwards without affecting the final results.
def binary_fraction(x: torch.Tensor, eps: float = 1e-3) -> float:
    dist = torch.minimum(torch.abs(x - 1.0), torch.abs(x + 1.0))
    return float((dist <= eps).float().mean().item())

# ==========================================================================================
# New local search: 1-flip to 1-opt, then 2-swap (A/B exchange)
# ==========================================================================================
@torch.no_grad()
def local_search_1flip_then_2swap_logged(
    x_bin: torch.Tensor,
    A: torch.Tensor,
    max_iters: int = 1_000_000,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, float, Dict[str, float]]:

    device = x_bin.device
    N = x_bin.numel()

    x = x_bin.clone()
    x = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    # v = A @ x
    v = A @ x

    # Precompute undirected edge list once: edges where A[i,j] > 0 with i < j
    edges = torch.nonzero(torch.triu(A, diagonal=1) > 0, as_tuple=False)  # (m,2)

    info = {
        "local_total_time": 0.0,
        "time_1flip": 0.0,
        "time_2swap": 0.0,
        "improve_1flip": 0.0,
        "improve_2swap": 0.0,
    }

    total_local_start = time.perf_counter()

    if edges.numel() == 0:
        info["local_total_time"] = time.perf_counter() - total_local_start
        return x, 0.0, info

    ei = edges[:, 0]
    ej = edges[:, 1]
    w = A[ei, ej]

    for _ in range(max_iters):
        # --------------------------------------------------------
        # Stage 1: best 1-flip
        # --------------------------------------------------------
        t1_start = time.perf_counter()

        gains = x * v
        max_gain, k = torch.max(gains, dim=0)

        if max_gain > tol:
            gain_val = float(max_gain.item())

            k = int(k.item())
            x[k] = -x[k]
            v += 2.0 * x[k] * A[:, k]

            info["improve_1flip"] += gain_val
            info["time_1flip"] += (time.perf_counter() - t1_start)
            continue

        info["time_1flip"] += (time.perf_counter() - t1_start)

        # --------------------------------------------------------
        # Stage 2: best 2-swap over edges only (neighbor-restricted)
        # --------------------------------------------------------
        t2_start = time.perf_counter()

        diff_side = (x[ei] * x[ej]) < 0
        if not bool(diff_side.any().item()):
            info["time_2swap"] += (time.perf_counter() - t2_start)
            break

        pair_gain = gains[ei] + gains[ej] + 2.0 * w
        pair_gain = torch.where(
            diff_side,
            pair_gain,
            torch.tensor(-float("inf"), device=device)
        )

        best_pair_gain, best_idx = torch.max(pair_gain, dim=0)
        if best_pair_gain <= tol:
            info["time_2swap"] += (time.perf_counter() - t2_start)
            break

        gain_val = float(best_pair_gain.item())

        t = int(best_idx.item())
        i = int(ei[t].item())
        j = int(ej[t].item())

        x[i] = -x[i]
        v += 2.0 * x[i] * A[:, i]

        x[j] = -x[j]
        v += 2.0 * x[j] * A[:, j]

        info["improve_2swap"] += gain_val
        info["time_2swap"] += (time.perf_counter() - t2_start)

    xAx = (x @ v).item()
    deg_sum = torch.sum(A).item()
    final_cut = 0.25 * (deg_sum - xAx)

    info["local_total_time"] = time.perf_counter() - total_local_start

    return x, float(final_cut), info

# ==========================================================================================
# Run all ER settings
# ==========================================================================================
for (n, p) in SETTINGS:
    p_tag = str(p).replace('.', 'p')

    print("\n" + "#"*80)
    print(f"### Running setting: n={n}, p={p}")
    print("#"*80)

    seed_best_list = []

    for graph_seed in SEEDS:
        print("\n" + "="*70)
        print(f"[Setting n={n}, p={p}] [Seed {graph_seed}] Start")
        print("="*70)

        # Generate an ER graph using NetworkX
        G = nx.gnp_random_graph(n, p, seed=graph_seed)
        num_components = nx.number_connected_components(G)
        m = G.number_of_edges()

        print("Number of connected components:", num_components)
        print("n =", n, "m =", m)

        # Build A, L matrix
        A = nx.adjacency_matrix(G).todense()
        A = torch.tensor(A, dtype=torch.float32, device=device)

        deg = torch.sum(A, dim=1)
        L = torch.diag(deg) - A

        # ones vector used by eps * 1^T x and eps * 1
        ones = torch.ones(n, device=device)

        sum_deg = float(deg.sum().item())
        print("sum_i d_i =", sum_deg, "(should be 2m =", 2*m, ")")

        # Degree-based initialization
        degrees = dict(G.degree())
        max_degree_node = max(degrees.values())

        d = torch.zeros(n, device=device)
        for node, degree_i in G.degree():
            d[node] = 1.0 - (degree_i / max_degree_node)
        d = d / d.max()
        d = 2.0 * d - 1.0

        exploration_parameter_eta = 0.5
        cov = exploration_parameter_eta * torch.eye(n, device=device)

        # Optimization loop
        results = []
        process = psutil.Process(os.getpid())
        start_time = time.time()

        global_best_cut = float("nan")
        global_best_xbin = None
        global_best_loss = float("nan")

        init = 0
        # Start timing
        while time.time() - start_time < TIME_LIMIT:

            init_start_time = time.time()

            if init == 0:
                x = d.clone()
            else:
                torch.manual_seed(init)
                x = torch.normal(mean=d, std=torch.sqrt(torch.diag(cov)))

            x = torch.clamp(x, -1.0, 1.0)

            # Initialize the momentum v
            v = torch.zeros(n, device=device)

            best_cut = float("nan")
            best_xbin = None
            best_loss = float("nan")
            best_x = None

            last_converged_iter = None
            resets_done = 0

            # ========================================================
            # Phase 1: Initial gradient descent convergence (rr = 0)
            # Phase 2: Global search (rr >= 1)
            # ========================================================
            gd_time = 0.0
            gd_cut = float("nan")

            global_search_time = 0.0
            global_search_cut = float("nan")

            gd_phase_done = False
            gd_phase_start = time.time()

            local_improve = 0.0
            local_total_time = 0.0
            time_1flip = 0.0
            time_2swap = 0.0
            improve_1flip = 0.0
            improve_2swap = 0.0

            for rr in range(RESET_ROUNDS):
                if time.time() - start_time >= TIME_LIMIT:
                    break

                rr_start_time = time.time()
                resets_done += 1

                round_best_cut = float("nan")
                round_best_xbin = None
                round_best_loss = float("nan")
                round_best_x = None

                last_converged_iter = None

                for it in range(ITERATION_T):
                    if time.time() - start_time >= TIME_LIMIT:
                        break

                    # ============================================================
                    # Use the gradient: f'_AL(x)=2Ax+eps*1
                    # ============================================================
                    g = grad_AL(x, A, EPS_LINEAR_BIAS, ones)
                    v = beta * v + alpha * g
                    x = torch.clamp(x - v, -1.0, 1.0)

                    if is_binary_converged(x, eps=EPS_BOUNDARY):
                        last_converged_iter = it + 1
                        x_bin = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

                        # Compute Max-Cut using Laplacian
                        cutv = cut_value_from_binary(x_bin, L)

                        # Loss should match the optimized objective f_AL for logging consistency
                        lv = float(loss_AL(x, A, EPS_LINEAR_BIAS, ones).item())

                        if rr == 0:
                            if (round_best_xbin is None) or (cutv > round_best_cut):
                                round_best_cut = cutv
                                round_best_xbin = x_bin.clone()
                                round_best_loss = lv
                                round_best_x = x.clone()
                        else:
                            round_best_cut = cutv
                            round_best_xbin = x_bin.clone()
                            round_best_loss = lv
                            round_best_x = x.clone()
                            break

                rr_elapsed = time.time() - rr_start_time

                if rr == 0 and (not gd_phase_done):
                    gd_time = time.time() - gd_phase_start
                    gd_cut = round_best_cut if round_best_xbin is not None else float("nan")
                    gd_phase_done = True

                if round_best_xbin is not None:
                    cut_before_local = round_best_cut

                    local_xbin, local_cut, local_info = local_search_1flip_then_2swap_logged(
                        round_best_xbin,
                        A,
                        max_iters=1_000_000,
                        tol=1e-6
                    )

                    local_improve += (local_cut - cut_before_local)
                    local_total_time += local_info["local_total_time"]
                    time_1flip += local_info["time_1flip"]
                    time_2swap += local_info["time_2swap"]
                    improve_1flip += local_info["improve_1flip"]
                    improve_2swap += local_info["improve_2swap"]

                    round_best_xbin = local_xbin.clone()
                    round_best_cut = local_cut
                    round_best_x = local_xbin.to(torch.float32).clone()
                    round_best_loss = float(
                        loss_AL(round_best_x, A, EPS_LINEAR_BIAS, ones).item()
                    )

                # accept or not
                if round_best_xbin is not None:
                    if (best_xbin is None) or (np.isnan(best_cut)) or (round_best_cut > best_cut):
                        best_cut = round_best_cut
                        best_xbin = round_best_xbin.clone()
                        best_loss = round_best_loss
                        best_x = round_best_x.clone() if (round_best_x is not None) else x.clone()
                        accepted_improved = True
                    else:
                        accepted_improved = False
                else:
                    accepted_improved = False

                # global reset
                if rr < RESET_ROUNDS - 1:
                    if time.time() - start_time >= TIME_LIMIT:
                        break

                    if accepted_improved and (round_best_x is not None):
                        base_x = round_best_x
                    else:
                        base_x = best_x if (best_x is not None) else x

                    k = int(RESET_FRAC * n)
                    idx = torch.randperm(n, device=device)[:k]

                    x = base_x.clone()
                    x[idx] = 0.0
                    x = torch.clamp(x, -1.0, 1.0)
                    v = torch.zeros(n, device=device)

                if rr >= 1:
                    global_search_time += rr_elapsed

            global_search_cut = best_cut if best_xbin is not None else float("nan")

            # update global best (within this seed)
            if best_xbin is not None:
                if (global_best_xbin is None) or (np.isnan(global_best_cut)) or (best_cut > global_best_cut):
                    global_best_cut = best_cut
                    global_best_xbin = best_xbin.clone()
                    global_best_loss = best_loss

            frac_bin = binary_fraction(x, eps=EPS_BOUNDARY)

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(device=device) / 1024**2
                gpu_max_mem = torch.cuda.max_memory_allocated(device=device) / 1024**2
            else:
                gpu_mem, gpu_max_mem = 0.0, 0.0

            cpu_mem = process.memory_info().rss / 1024**2

            print(
                f"[n={n},p={p},seed={graph_seed}] init={init:02d} "
                f"gd_time={gd_time:.2f}s gd_cut={gd_cut} "
                f"global_search_time={global_search_time:.2f}s global_search_cut={global_search_cut} "
                f"1flip_time={time_1flip:.2f}s 1flip_improve={improve_1flip:.1f} "
                f"2swap_time={time_2swap:.2f}s 2swap_improve={improve_2swap:.1f} "
                f"best_cut={best_cut} loss≈{best_loss} "
                f"last_converged_iter={last_converged_iter} "
                f"resets_done={resets_done}/{RESET_ROUNDS} "
                f"bin_frac={frac_bin:.3f} "
                f"local_improve={local_improve:.1f} "
                f"global_best_cut={global_best_cut}"
            )

            results.append((
                init,
                best_cut,
                best_loss,
                last_converged_iter,
                resets_done,
                frac_bin,
                local_improve,
                cpu_mem,
                gpu_mem,
                gpu_max_mem,
                gd_time,
                gd_cut,
                global_search_time,
                global_search_cut,
                local_total_time,
                time_1flip,
                improve_1flip,
                time_2swap,
                improve_2swap
            ))
            init += 1

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / max(init, 1)

        # ------------------------------------------------------------
        # Save file 1: intermediate log for this seed+setting
        # ------------------------------------------------------------
        inter_filename = f"{n}_GPU_MaxCut_ER_p{p_tag}_seed{graph_seed}_intermediate.txt"

        with open(inter_filename, "w") as f:
            f.write(f"device = {device}\n\n")
            f.write(f"n = {n}\n")
            f.write(f"p = {p}\n")
            f.write(f"graph_seed = {graph_seed}\n")
            f.write(f"m = {m}\n")
            f.write(f"sum_deg = {sum_deg}\n\n")

            f.write(f"alpha = {alpha}\n")
            f.write(f"beta = {beta}\n")
            f.write(f"ITERATION_T = {ITERATION_T}\n")
            f.write(f"RESET_ROUNDS = {RESET_ROUNDS}\n")
            f.write(f"RESET_FRAC = {RESET_FRAC}\n")
            f.write(f"EPS_BOUNDARY = {EPS_BOUNDARY}\n")
            f.write(f"EPS_LINEAR_BIAS = {EPS_LINEAR_BIAS}\n")
            f.write(f"TIME_LIMIT = {TIME_LIMIT}\n\n")

            f.write("Results per init:\n")
            for (
                init_,
                best_cut_,
                best_loss_,
                conv_it,
                resets_done_,
                frac_bin_,
                local_improve_,
                cpu_mem_,
                gpu_mem_,
                gpu_max_mem_,
                gd_time_,
                gd_cut_,
                global_search_time_,
                global_search_cut_,
                local_total_time_,
                time_1flip_,
                improve_1flip_,
                time_2swap_,
                improve_2swap_
            ) in results:
                f.write(
                    f"n={n}, p={p}, seed={graph_seed}, init={init_}, "
                    f"gd_time={gd_time_:.4f}s, gd_cut={gd_cut_}, "
                    f"global_search_time={global_search_time_:.4f}s, global_search_cut={global_search_cut_}, "
                    f"local_total_time={local_total_time_:.4f}s, "
                    f"1flip_time={time_1flip_:.4f}s, 1flip_improve={improve_1flip_:.4f}, "
                    f"2swap_time={time_2swap_:.4f}s, 2swap_improve={improve_2swap_:.4f}, "
                    f"best_cut={best_cut_}, best_loss≈{best_loss_}, "
                    f"last_converged_iter={conv_it}, resets_done={resets_done_}, "
                    f"bin_frac={frac_bin_:.4f}, local_improve={local_improve_:.4f}, "
                    f"CPU={cpu_mem_:.2f}MB, GPU={gpu_mem_:.2f}MB, GPUpeak={gpu_max_mem_:.2f}MB\n"
                )

            cuts = np.array([r[1] for r in results], dtype=np.float64)
            cuts_valid = cuts[~np.isnan(cuts)]

            f.write("\nSummary (this seed):\n")
            if cuts_valid.size > 0:
                f.write(f"best_over_inits = {np.max(cuts_valid):.1f}\n")
                f.write(f"avg_over_valid_inits = {np.mean(cuts_valid):.3f}\n")
                f.write(f"std_over_valid_inits = {np.std(cuts_valid):.3f}\n")
                f.write(f"valid_runs = {cuts_valid.size}/{len(cuts)}\n")
            else:
                f.write("No valid runs (no full boundary convergence).\n")

            f.write("\nTime:\n")
            f.write(f"total_time = {total_time:.2f} s\n")
            f.write(f"avg_time_per_init = {avg_time:.2f} s\n")

            f.write("\nGlobal best (this seed):\n")
            f.write(f"global_best_cut = {global_best_cut}\n")
            f.write(f"global_best_loss = {global_best_loss}\n")

        print(f"[n={n}, p={p}] [Seed {graph_seed}] Saved intermediate log: {inter_filename}")

        seed_best_list.append(float(global_best_cut) if not np.isnan(global_best_cut) else np.nan)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ============================================================
    # Save file 2: seed summary only for THIS (n,p)
    # ============================================================
    summary_filename = f"{n}_GPU_MaxCut_ER_p{p_tag}_seeds8_summary.txt"

    seed_best_arr = np.array(seed_best_list, dtype=np.float64)
    valid_seed_best = seed_best_arr[~np.isnan(seed_best_arr)]
    mean_best = float(np.mean(valid_seed_best)) if valid_seed_best.size > 0 else float("nan")

    with open(summary_filename, "w") as f:
        f.write(f"device = {device}\n\n")
        f.write(f"Setting: n={n}, p={p}\n")
        f.write(f"Seeds = {SEEDS}\n\n")

        f.write("Best cut per seed:\n")
        for s, val in zip(SEEDS, seed_best_list):
            f.write(f"seed={s}, best_cut={val}\n")

        f.write("\nAverage of best cuts over seeds:\n")
        f.write(f"mean_best_cut = {mean_best}\n")
        if valid_seed_best.size > 1:
            f.write(f"std_best_cut  = {float(np.std(valid_seed_best))}\n")
        f.write(f"valid_seeds = {valid_seed_best.size}/{len(SEEDS)}\n")

    print(f"[n={n}, p={p}] Saved seed summary: {summary_filename}")

print("\nDone. All settings finished.")
