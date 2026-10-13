import torch
import networkx as nx
import numpy as np
import psutil, os
import time
import scipy.sparse as sp
from typing import Tuple, Dict, List

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
    (1000, 0.5),
    (1000, 0.1),
    (100,  0.5),
]

SEEDS = list(range(1))  # [0,1,2,3,4,5,6,7]

# ============================================================
# Parallel parameters (borrowed from MIS logic)
# ============================================================
BATCH_SIZE = 512
TOP_K = 5
CHECK_EVERY = 50

# ==========================================================================================
# Hyperparameters
# Hyperparameter settings differ for large graphs and small graphs.
# ==========================================================================================
alpha = 0.0025
beta  = 0.8
ITERATION_T = 3000

RESET_ROUNDS = 90
RESET_FRAC = 0.8
EPS_BOUNDARY = 0.4
TIME_LIMIT = 10 * 60  # per seed

# linear bias epsilon for tie-breaking (0 < eps << 1)
EPS_LINEAR_BIAS = 1e-8

# ==========================================================================================
# Sparse helpers
# ==========================================================================================
def scipy_csr_to_torch_csr(A_scipy: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    A_scipy = A_scipy.tocsr().astype(np.float32)
    indptr = torch.tensor(A_scipy.indptr, dtype=torch.int64, device=device)
    indices = torch.tensor(A_scipy.indices, dtype=torch.int64, device=device)
    data = torch.tensor(A_scipy.data, dtype=torch.float32, device=device)
    A_torch = torch.sparse_csr_tensor(
        indptr, indices, data,
        size=A_scipy.shape,
        device=device
    )
    return A_torch

def spmv(A_csr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(A_csr, x.unsqueeze(1)).squeeze(1)

def spmm_dense(A_csr: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # X: (n, B)
    return torch.sparse.mm(A_csr, X)

# ==========================================================================================
# Objective / Gradient
# ==========================================================================================
def loss_AL(x: torch.Tensor, A: torch.Tensor, eps: float, ones: torch.Tensor) -> torch.Tensor:
    Ax = spmv(A, x)
    return (x @ Ax) + eps * (ones @ x)

def grad_AL(x: torch.Tensor, A: torch.Tensor, eps: float, ones: torch.Tensor) -> torch.Tensor:
    Ax = spmv(A, x)
    return 2.0 * Ax + eps * ones

def cut_value_from_binary(x_bin: torch.Tensor, edge_u: torch.Tensor, edge_v: torch.Tensor, edge_w: torch.Tensor, total_edge_weight: float) -> float:
    # cut(x_bin) = sum_{(i,j) in E} w_ij * (1 - x_i x_j) / 2
    prod = x_bin[edge_u] * x_bin[edge_v]
    return float(0.5 * (total_edge_weight - torch.sum(edge_w * prod).item()))

def cut_values_from_binary_batch(X_bin: torch.Tensor, edge_u: torch.Tensor, edge_v: torch.Tensor, edge_w: torch.Tensor, total_edge_weight: float) -> torch.Tensor:
    # X_bin: (n, B)
    prod = X_bin[edge_u, :] * X_bin[edge_v, :]
    cuts = 0.5 * (total_edge_weight - torch.sum(edge_w.unsqueeze(1) * prod, dim=0))
    return cuts

def is_binary_converged(x: torch.Tensor, eps: float = 1e-3) -> bool:
    dist = torch.minimum(torch.abs(x - 1.0), torch.abs(x + 1.0))
    return bool(torch.all(dist <= eps).item())

def binary_fraction(x: torch.Tensor, eps: float = 1e-3) -> float:
    dist = torch.minimum(torch.abs(x - 1.0), torch.abs(x + 1.0))
    return float((dist <= eps).float().mean().item())

def binary_fraction_batch(X: torch.Tensor, eps: float = 1e-3) -> float:
    dist = torch.minimum(torch.abs(X - 1.0), torch.abs(X + 1.0))
    return float((dist <= eps).float().mean().item())

def boundary_converged_mask_batch(X: torch.Tensor, eps: float) -> torch.Tensor:
    dist = torch.minimum(torch.abs(X - 1.0), torch.abs(X + 1.0))
    return (dist <= eps).all(dim=0)


@torch.no_grad()
def local_search_1flip_then_2swap_logged(
    x_bin: torch.Tensor,
    A: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    total_edge_weight: float,
    max_iters: int = 1_000_000,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, float, Dict[str, float]]:

    device = x_bin.device
    x = x_bin.clone()
    x = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    crow = A.crow_indices()
    col = A.col_indices()
    val = A.values()

    # v = A @ x
    v = spmv(A, x)

    info = {
        "local_total_time": 0.0,
        "time_1flip": 0.0,
        "time_2swap": 0.0,
        "improve_1flip": 0.0,
        "improve_2swap": 0.0,
    }

    total_local_start = time.perf_counter()

    if edge_u.numel() == 0:
        info["local_total_time"] = time.perf_counter() - total_local_start
        return x, 0.0, info

    ei = edge_u
    ej = edge_v
    w = edge_w
    neg_inf = torch.tensor(-float("inf"), device=device)

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

            row_start = int(crow[k].item())
            row_end = int(crow[k + 1].item())
            nbrs = col[row_start:row_end]
            nbrw = val[row_start:row_end]

            if nbrs.numel() > 0:
                v[nbrs] += 2.0 * x[k] * nbrw

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
        pair_gain = torch.where(diff_side, pair_gain, neg_inf)

        best_pair_gain, best_idx = torch.max(pair_gain, dim=0)
        if best_pair_gain <= tol:
            info["time_2swap"] += (time.perf_counter() - t2_start)
            break

        gain_val = float(best_pair_gain.item())

        t = int(best_idx.item())
        i = int(ei[t].item())
        j = int(ej[t].item())

        x[i] = -x[i]
        row_start_i = int(crow[i].item())
        row_end_i = int(crow[i + 1].item())
        nbrs_i = col[row_start_i:row_end_i]
        nbrw_i = val[row_start_i:row_end_i]
        if nbrs_i.numel() > 0:
            v[nbrs_i] += 2.0 * x[i] * nbrw_i

        x[j] = -x[j]
        row_start_j = int(crow[j].item())
        row_end_j = int(crow[j + 1].item())
        nbrs_j = col[row_start_j:row_end_j]
        nbrw_j = val[row_start_j:row_end_j]
        if nbrs_j.numel() > 0:
            v[nbrs_j] += 2.0 * x[j] * nbrw_j

        info["improve_2swap"] += gain_val
        info["time_2swap"] += (time.perf_counter() - t2_start)

    final_cut = cut_value_from_binary(x, edge_u, edge_v, edge_w, total_edge_weight)
    info["local_total_time"] = time.perf_counter() - total_local_start

    return x, float(final_cut), info

# ==========================================================================================
# Parallel helpers (borrowed from MIS logic and adapted to MaxCut)
# ==========================================================================================
def init_parallel_population(d_base: torch.Tensor, n: int, device: torch.device, exploration_eta: float):
    noise = torch.randn((n, BATCH_SIZE), device=device) * float(np.sqrt(exploration_eta))
    return torch.clamp(d_base.unsqueeze(1).expand(n, BATCH_SIZE) + noise, -1.0, 1.0)

def extract_topk_from_batch_by_cut(cuts: torch.Tensor, X_bin: torch.Tensor, top_k: int):
    """
    cuts: (B,)
    X_bin: (n, B)
    return: List[Tuple[float, torch.Tensor]]
    """
    if cuts.numel() == 0:
        return []

    k_eff = min(top_k, cuts.numel())
    vals, idxs = torch.topk(cuts, k=k_eff)

    sols = []
    for j in range(k_eff):
        col_idx = int(idxs[j].item())
        xcol = X_bin[:, col_idx].detach().clone()
        sols.append((float(vals[j].item()), xcol))
    return sols

def merge_keep_topk(old_topk, new_topk, top_k: int):
    merged = old_topk + new_topk
    if len(merged) == 0:
        return []
    merged.sort(key=lambda x: x[0], reverse=True)
    return merged[:top_k]

def run_parallel_gd_until_converge(
    input_tensor: torch.Tensor,
    adjacency_matrix_tensor: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    total_edge_weight: float,
    device: torch.device,
    BETA: float,
    ALPHA: float,
    ITERATION_T: int,
    top_k: int,
    eps_linear_bias: float,
    eps_boundary: float,
    ones: torch.Tensor,
):
    """
    Returns:
        topk_solutions
        best_cut_seen_in_this_phase
        best_X_bin_seen_in_this_phase
        last_converged_iter
        frac_bin
    """
    n, B = input_tensor.shape
    input_velocity = torch.zeros((n, B), device=device)

    best_cut_seen = None
    best_xbin_seen = None
    last_converged_iter = None

    for i in range(ITERATION_T):
        Ax = spmm_dense(adjacency_matrix_tensor, input_tensor)
        gradient = 2.0 * Ax + eps_linear_bias * ones.unsqueeze(1)

        input_velocity = BETA * input_velocity + ALPHA * gradient
        input_tensor = torch.clamp(input_tensor - input_velocity, -1.0, 1.0)

        if (i + 1) % CHECK_EVERY == 0:
            converged_mask = boundary_converged_mask_batch(input_tensor, eps_boundary)
            X_bin = torch.where(input_tensor >= 0, torch.ones_like(input_tensor), -torch.ones_like(input_tensor))
            cuts = cut_values_from_binary_batch(X_bin, edge_u, edge_v, edge_w, total_edge_weight)

            current_best_val, current_best_idx = torch.max(cuts, dim=0)
            if (best_cut_seen is None) or (float(current_best_val.item()) > best_cut_seen):
                best_cut_seen = float(current_best_val.item())
                best_xbin_seen = X_bin[:, int(current_best_idx.item())].detach().clone()

            if converged_mask.any():
                last_converged_iter = i + 1
                valid_cuts = torch.where(converged_mask, cuts, torch.full_like(cuts, -float("inf")))
                return (
                    extract_topk_from_batch_by_cut(valid_cuts, X_bin, top_k),
                    best_cut_seen,
                    best_xbin_seen,
                    last_converged_iter,
                    binary_fraction_batch(input_tensor, eps_boundary),
                )

    X_bin = torch.where(input_tensor >= 0, torch.ones_like(input_tensor), -torch.ones_like(input_tensor))
    cuts = cut_values_from_binary_batch(X_bin, edge_u, edge_v, edge_w, total_edge_weight)

    current_best_val, current_best_idx = torch.max(cuts, dim=0)
    if (best_cut_seen is None) or (float(current_best_val.item()) > best_cut_seen):
        best_cut_seen = float(current_best_val.item())
        best_xbin_seen = X_bin[:, int(current_best_idx.item())].detach().clone()

    return [], best_cut_seen, best_xbin_seen, last_converged_iter, binary_fraction_batch(input_tensor, eps_boundary)

def build_parallel_resets_from_topk(
    topk_solutions,
    n: int,
    device: torch.device,
    RESET_RATIO: float,
):
    reset_tensor = torch.zeros((n, BATCH_SIZE), device=device)

    if len(topk_solutions) == 0:
        return reset_tensor

    for b in range(BATCH_SIZE):
        parent_cut, parent_x = topk_solutions[np.random.randint(len(topk_solutions))]
        parent_x = parent_x.to(device)

        keep_mask = torch.ones(n, device=device)
        k = int(RESET_RATIO * n)
        if k > 0:
            idx = torch.randperm(n, device=device)[:k]
            keep_mask[idx] = 0.0

        reset_tensor[:, b] = parent_x * keep_mask

    return torch.clamp(reset_tensor, -1.0, 1.0)

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

        # Build sparse A and edge list
        A_scipy = nx.adjacency_matrix(G).tocsr().astype(np.float32)
        A = scipy_csr_to_torch_csr(A_scipy, device)

        deg_np = np.asarray(A_scipy.sum(axis=1)).reshape(-1).astype(np.float32)
        deg = torch.tensor(deg_np, dtype=torch.float32, device=device)

        upper = sp.triu(A_scipy, k=1).tocoo()
        edge_u = torch.tensor(upper.row, dtype=torch.int64, device=device)
        edge_v = torch.tensor(upper.col, dtype=torch.int64, device=device)
        edge_w = torch.tensor(upper.data, dtype=torch.float32, device=device)

        # ones vector used by eps * 1^T x and eps * 1
        ones = torch.ones(n, device=device)

        sum_deg = float(deg.sum().item())
        total_edge_weight = float(edge_w.sum().item())
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

            best_cut = float("nan")
            best_xbin = None
            best_loss = float("nan")
            best_x = None

            last_converged_iter = None
            resets_done = 0

            # ========================================================
            # Phase 1: Initial parallel gradient descent
            # Phase 2: Global search with parallel reset
            # ========================================================
            gd_time = 0.0
            gd_cut = float("nan")

            global_search_time = 0.0
            global_search_cut = float("nan")

            local_improve = 0.0
            local_total_time = 0.0
            time_1flip = 0.0
            time_2swap = 0.0
            improve_1flip = 0.0
            improve_2swap = 0.0

            # ---------------- Phase 1 ----------------
            gd_phase_start = time.time()
            input_tensor = init_parallel_population(d, n, device, exploration_parameter_eta)

            topk_solutions, best_cut_seen_phase1, best_xbin_seen_phase1, conv_it_phase1, frac_bin = run_parallel_gd_until_converge(
                input_tensor=input_tensor,
                adjacency_matrix_tensor=A,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                total_edge_weight=total_edge_weight,
                device=device,
                BETA=beta,
                ALPHA=alpha,
                ITERATION_T=ITERATION_T,
                top_k=TOP_K,
                eps_linear_bias=EPS_LINEAR_BIAS,
                eps_boundary=EPS_BOUNDARY,
                ones=ones,
            )

            gd_time = time.time() - gd_phase_start
            gd_cut = best_cut_seen_phase1 if best_cut_seen_phase1 is not None else float("nan")
            last_converged_iter = conv_it_phase1

            if len(topk_solutions) > 0:
                best_cut = topk_solutions[0][0]
                best_xbin = topk_solutions[0][1].clone()
                best_x = best_xbin.to(torch.float32).clone()
                best_loss = float(loss_AL(best_x, A, EPS_LINEAR_BIAS, ones).item())
            elif best_xbin_seen_phase1 is not None:
                best_cut = best_cut_seen_phase1
                best_xbin = best_xbin_seen_phase1.clone()
                best_x = best_xbin.to(torch.float32).clone()
                best_loss = float(loss_AL(best_x, A, EPS_LINEAR_BIAS, ones).item())

            # ---------------- Phase 2 ----------------
            for rr in range(1, RESET_ROUNDS):
                if time.time() - start_time >= TIME_LIMIT:
                    break

                rr_start_time = time.time()
                resets_done += 1

                input_tensor = build_parallel_resets_from_topk(
                    topk_solutions=topk_solutions if len(topk_solutions) > 0 else [(best_cut, best_xbin)],
                    n=n,
                    device=device,
                    RESET_RATIO=RESET_FRAC,
                )

                new_topk, best_cut_seen_phase2, best_xbin_seen_phase2, conv_it_phase2, frac_bin = run_parallel_gd_until_converge(
                    input_tensor=input_tensor,
                    adjacency_matrix_tensor=A,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    total_edge_weight=total_edge_weight,
                    device=device,
                    BETA=beta,
                    ALPHA=alpha,
                    ITERATION_T=ITERATION_T,
                    top_k=TOP_K,
                    eps_linear_bias=EPS_LINEAR_BIAS,
                    eps_boundary=EPS_BOUNDARY,
                    ones=ones,
                )

                if len(new_topk) > 0:
                    topk_solutions = merge_keep_topk(topk_solutions, new_topk, TOP_K)
                    if topk_solutions[0][0] > best_cut:
                        best_cut = topk_solutions[0][0]
                        best_xbin = topk_solutions[0][1].clone()
                        best_x = best_xbin.to(torch.float32).clone()
                        best_loss = float(loss_AL(best_x, A, EPS_LINEAR_BIAS, ones).item())
                elif best_xbin_seen_phase2 is not None and ((np.isnan(best_cut)) or (best_cut_seen_phase2 > best_cut)):
                    best_cut = best_cut_seen_phase2
                    best_xbin = best_xbin_seen_phase2.clone()
                    best_x = best_xbin.to(torch.float32).clone()
                    best_loss = float(loss_AL(best_x, A, EPS_LINEAR_BIAS, ones).item())

                global_search_time += (time.time() - rr_start_time)

            global_search_cut = best_cut if best_xbin is not None else float("nan")

            # ---------------- Local Search on final top-k ----------------
            if len(topk_solutions) == 0 and best_xbin is not None:
                topk_solutions = [(best_cut, best_xbin.clone())]

            final_best_cut_after_local = best_cut
            final_best_xbin_after_local = best_xbin.clone() if best_xbin is not None else None

            for cand_cut, cand_xbin in topk_solutions:
                cut_before_local = cand_cut

                local_xbin, local_cut, local_info = local_search_1flip_then_2swap_logged(
                    cand_xbin,
                    A,
                    edge_u,
                    edge_v,
                    edge_w,
                    total_edge_weight,
                    max_iters=1_000_000,
                    tol=1e-6
                )

                local_improve += (local_cut - cut_before_local)
                local_total_time += local_info["local_total_time"]
                time_1flip += local_info["time_1flip"]
                time_2swap += local_info["time_2swap"]
                improve_1flip += local_info["improve_1flip"]
                improve_2swap += local_info["improve_2swap"]

                if local_cut > final_best_cut_after_local:
                    final_best_cut_after_local = local_cut
                    final_best_xbin_after_local = local_xbin.clone()

            best_cut = final_best_cut_after_local
            best_xbin = final_best_xbin_after_local
            if best_xbin is not None:
                best_x = best_xbin.to(torch.float32).clone()
                best_loss = float(loss_AL(best_x, A, EPS_LINEAR_BIAS, ones).item())

            # update global best (within this seed)
            if best_xbin is not None:
                if (global_best_xbin is None) or (np.isnan(global_best_cut)) or (best_cut > global_best_cut):
                    global_best_cut = best_cut
                    global_best_xbin = best_xbin.clone()
                    global_best_loss = best_loss

            if best_x is not None:
                frac_bin = binary_fraction(best_x, eps=EPS_BOUNDARY)
            else:
                frac_bin = 0.0

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
            f.write(f"TIME_LIMIT = {TIME_LIMIT}\n")
            f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
            f.write(f"TOP_K = {TOP_K}\n")
            f.write(f"CHECK_EVERY = {CHECK_EVERY}\n\n")

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
