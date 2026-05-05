
import os
import time
import random
import datetime
from pathlib import Path
from typing import List, Iterable, Set, Optional, Tuple

import torch
import networkx as nx
import numpy as np

# ============================ device ============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================ parallel parameters ============================
BATCH_SIZE = 512
TOP_K = 5          
CHECK_EVERY = 50   # batch checker frequency

class TeeLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.f = open(filepath, "w", encoding="utf-8", buffering=1)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def print(self, *args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args) + end
        print(*args, sep=sep, end=end)
        self.f.write(msg)

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================ parameter ============================
def get_params_for_graph(n: int, p: float):
    group_1 = {
        (1000, 0.10010),
        (3000, 0.03334),
    }

    group_2 = {
        (3000, 0.33344),
        (3000, 0.10003),
        (1000, 0.30030),
        (1000, 0.50050),
    }
    key = (n, round(p, 5))

    if key in group_1:
        return 0.30, 0.80
    elif key in group_2:
        return 0.45, 0.80
    else:
        return 0.75, 0.80


def make_graph_dir(root_dir: str, n: int, p: float, beta: float, alpha: float) -> str:
    p_str = f"{p:.5f}".replace(".", "p")
    beta_str = f"{beta:.2f}".replace(".", "p")
    alpha_str = f"{alpha:.2f}".replace(".", "p")
    d = Path(root_dir) / f"n{n}_p{p_str}_beta{beta_str}_alpha{alpha_str}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

# ============================ MIS Checker ============================
def MIS_checker_efficient_3(X_torch, adjacency_matrix_tensor):
    n = X_torch.shape[0]
    # binarized X
    X_torch_binarized = X_torch.bool().float()
    # if for some gradient update, we are still at the boundary, then we have maximal IS
    X_torch_binarized_update = X_torch_binarized - 0.1 * (
        -torch.ones(n, device=X_torch.device) + (n * adjacency_matrix_tensor) @ X_torch_binarized
    )
    # Projection to [0,1]
    X_torch_binarized_update[X_torch_binarized_update >= 1] = 1
    X_torch_binarized_update[X_torch_binarized_update <= 0] = 0
    if torch.equal(X_torch_binarized, X_torch_binarized_update):
        # we have a maximal IS:
        MIS = torch.nonzero(X_torch_binarized).squeeze()
        # Exit the function with True
        return True, MIS
    return False, None


@torch.no_grad()
def MIS_checker_batch(X_batch: torch.Tensor, adjacency_matrix_tensor: torch.Tensor):
    n, B = X_batch.shape

    X_torch_binarized = X_batch.bool().float()  # (n, B)
    X_torch_binarized_update = X_torch_binarized - 0.1 * (
        -torch.ones(n, device=X_batch.device).unsqueeze(1) +
        (n * adjacency_matrix_tensor) @ X_torch_binarized
    )
    X_torch_binarized_update[X_torch_binarized_update >= 1] = 1
    X_torch_binarized_update[X_torch_binarized_update <= 0] = 0

    converged_mask = torch.eq(X_torch_binarized, X_torch_binarized_update).all(dim=0)  # (B,)
    sizes = X_torch_binarized.sum(dim=0) * converged_mask.float()  # (B,)

    return converged_mask, sizes, X_torch_binarized


# ============================ 1-2 Swap (CPU) ============================
def mis_local_improve_12(
    adj: List[Iterable[int]],
    M_init: Iterable[int],
    max_outer_loops: int = 50,
    pair_checks_per_u: int = 400,
    rng: Optional[random.Random] = None
) -> Set[int]:
    if rng is None:
        rng = random.Random(0)

    n = len(adj)
    adjS = [set(N) if not isinstance(N, set) else N for N in adj]
    M = set(M_init)
    inM = [False] * n
    for u in M:
        inM[u] = True

    degM = [0] * n
    for v in range(n):
        cnt = 0
        for w in adjS[v]:
            if inM[w]:
                cnt += 1
        degM[v] = cnt

    def recompute_onlyNbr(v: int) -> int:
        if degM[v] != 1:
            return -1
        for w in adjS[v]:
            if inM[w]:
                return w
        return -1

    onlyNbrInM = [-1] * n
    for v in range(n):
        onlyNbrInM[v] = recompute_onlyNbr(v)

    def add_to_M(x: int):
        inM[x] = True
        M.add(x)
        for w in adjS[x]:
            degM[w] += 1
            if degM[w] == 1:
                onlyNbrInM[w] = x
            else:
                onlyNbrInM[w] = -1

    def remove_from_M(x: int):
        inM[x] = False
        M.remove(x)
        for w in adjS[x]:
            degM[w] -= 1
            if degM[w] == 1:
                onlyNbrInM[w] = recompute_onlyNbr(w)
            else:
                onlyNbrInM[w] = -1

    def greedy_repair_to_maximal():
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if (not inM[v]) and degM[v] == 0:
                    add_to_M(v)
                    changed = True

    greedy_repair_to_maximal()

    def try_one_12_swap() -> bool:
        for u in list(M):
            Cu = [v for v in adjS[u] if (not inM[v]) and (onlyNbrInM[v] == u)]
            if len(Cu) < 2:
                continue

            total_pairs = len(Cu) * (len(Cu) - 1) // 2
            if total_pairs <= pair_checks_per_u:
                L = Cu
                for i in range(len(L)):
                    for j in range(i + 1, len(L)):
                        a, b = L[i], L[j]
                        if b in adjS[a]:
                            continue
                        remove_from_M(u)
                        add_to_M(a)
                        add_to_M(b)
                        greedy_repair_to_maximal()
                        return True
            else:
                anchors = rng.sample(Cu, min(len(Cu), max(8, int(pair_checks_per_u ** 0.5))))
                budget_per_anchor = max(1, pair_checks_per_u // max(1, len(anchors)))
                for a in anchors:
                    tries = 0
                    while tries < budget_per_anchor:
                        b = Cu[rng.randrange(len(Cu))]
                        if b == a or (b in adjS[a]):
                            tries += 1
                            continue
                        remove_from_M(u)
                        add_to_M(a)
                        add_to_M(b)
                        greedy_repair_to_maximal()
                        return True
        return False

    for _ in range(max_outer_loops):
        if not try_one_12_swap():
            break

    return M


# ============================ strict MIS checker ============================
def strict_mis_feasibility_check(G_nx: nx.Graph, M_nodes: Iterable[int]) -> bool:
    M_list = list(M_nodes)
    if len(M_list) <= 1:
        return True
    return G_nx.subgraph(M_list).number_of_edges() == 0


# ============================ 1-2 Swap adapter ============================
def one_two_swap_adapter_from_adj(
    adj_list_of_sets,
    G_nx,
    initial_mis_tensor,
    epoch_num,
    log: Optional[TeeLogger] = None
):
    def _p(*args, **kwargs):
        if log is None:
            print(*args, **kwargs)
        else:
            log.print(*args, **kwargs)

    if initial_mis_tensor is None or initial_mis_tensor.numel() == 0:
        return 0, set()

    M_init = set(initial_mis_tensor.detach().cpu().numpy().tolist())
    initial_size = len(M_init)
    _p(f"    [1-2 Swap]: 启动... 初始大小 {initial_size}")

    local_rng = random.Random(epoch_num)

    t0 = time.time()
    M_improved = mis_local_improve_12(
        adj=adj_list_of_sets,
        M_init=M_init,
        max_outer_loops=5000,
        pair_checks_per_u=600,
        rng=local_rng
    )
    t1 = time.time()

    final_size = len(M_improved)
    _p(f"    [1-2 Swap]: 完成。耗时 {t1 - t0:.4f}s。最终大小: {final_size} (+{final_size - initial_size})")
    return final_size, M_improved


# ============================ parallel helpers ============================
def init_parallel_population(d_base: torch.Tensor, n: int, device: torch.device, epoch: int, exploration_eta: float):
    if epoch == 0:
        return d_base.unsqueeze(1).expand(n, BATCH_SIZE).clone()
    else:
        noise = torch.randn((n, BATCH_SIZE), device=device) * float(np.sqrt(exploration_eta))
        return torch.clamp(d_base.unsqueeze(1).expand(n, BATCH_SIZE) + noise, 0.0, 1.0)


def extract_topk_from_batch(sizes: torch.Tensor, X_bin: torch.Tensor, top_k: int):
    if sizes.numel() == 0:
        return []

    k_eff = min(top_k, sizes.numel())
    vals, idxs = torch.topk(sizes, k=k_eff)

    sols = []
    for j in range(k_eff):
        if vals[j].item() > 0:
            col_idx = int(idxs[j].item())
            nodes = torch.nonzero(X_bin[:, col_idx], as_tuple=False).flatten().detach().clone()
            sols.append((int(vals[j].item()), nodes))
    return sols


def merge_keep_topk(old_topk, new_topk, top_k: int):
    merged = old_topk + new_topk
    if len(merged) == 0:
        return []

    merged.sort(key=lambda x: x[0], reverse=True)
    return merged[:top_k]


def run_parallel_mgd_until_converge(
    input_tensor: torch.Tensor,
    adjacency_matrix_tensor: torch.Tensor,
    device: torch.device,
    GAMMA: float,
    BETA: float,
    ALPHA: float,
    ITERATION_T: int,
    top_k: int,
):

    n, B = input_tensor.shape
    input_velocity = torch.zeros((n, B), device=device)

    for i in range(ITERATION_T):
        Ax = adjacency_matrix_tensor @ input_tensor
        gradient = -torch.ones((n, 1), device=device) + GAMMA * Ax

        input_velocity = BETA * input_velocity + ALPHA * gradient
        input_tensor = torch.clamp(input_tensor - input_velocity, 0, 1)

        if (i + 1) % CHECK_EVERY == 0:
            converged_mask, sizes, X_bin = MIS_checker_batch(input_tensor, adjacency_matrix_tensor)
            if converged_mask.any():
                return extract_topk_from_batch(sizes, X_bin, top_k)

    return []


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
        parent_size, parent_nodes = random.choice(topk_solutions)
        parent_nodes = parent_nodes.to(device)

        m = int(parent_nodes.numel())
        if m == 0:
            continue

        num_kick = int(m * RESET_RATIO)

        keep_mask = torch.ones(m, device=device)
        if num_kick > 0:
            shuffle_idx = torch.randperm(m, device=device)
            kick_pos = shuffle_idx[:num_kick]
            keep_mask[kick_pos] = 0.0

        reset_tensor[parent_nodes, b] = keep_mask

    return reset_tensor


# ============================ epoch ============================
def run_one_graph_one_seed(
    n: int,
    p: float,
    seed: int,
    device: torch.device,
    time_limit_sec: int,
    out_dir: str,
):
    seed_txt = os.path.join(out_dir, f"seed{seed}.txt")
    log = TeeLogger(seed_txt)

    try:
        log.print("==========================================")
        log.print(f"Run start: {now_str()}")
        log.print(f"(n, p, seed) = ({n}, {p}, {seed})")
        log.print(f"Device: {device}")
        log.print(f"Time limit: {time_limit_sec} sec")
        log.print(f"Batch size: {BATCH_SIZE}")
        log.print(f"Top-k: {TOP_K}")
        log.print("==========================================")

        log.print(f"Generating Graph (n={n}, p={p})... 这可能需要一点时间...")
        start_gen = time.time()

        try:
            G = nx.gnp_random_graph(n, p, seed=seed)
        except Exception as e:
            log.print(f"Error generating graph: {e}")
            raise

        adj_list_of_sets = [set(G.adj[i]) for i in range(n)]

        log.print("Moving to GPU...")
        adjacency_matrix_dense = nx.adjacency_matrix(G).todense()
        adjacency_matrix_tensor = torch.tensor(adjacency_matrix_dense, dtype=torch.float32).to(device)

        degrees = dict(G.degree())
        max_degree_node = max(degrees.values()) if degrees else 1.0
        d_cpu = torch.zeros(n)
        for node, degree in G.degree():
            d_cpu[node] = 1 - (degree / max_degree_node)

        if (d_cpu == 0).all():
            d_cpu = torch.full((n,), 0.5)
        else:
            d_cpu = d_cpu / max(d_cpu)

        d_base = d_cpu.to(device)

        log.print(f"Graph Ready! Total setup time: {time.time() - start_gen:.2f}s")
        log.print(f"Components: {nx.number_connected_components(G)}")

        GAMMA = 1.01
        BETA, ALPHA = get_params_for_graph(n, p)

        ITERATION_T = 4000
        NUM_RESETS = 60
        RESET_RATIO = 0.7
        TIME_LIMIT = time_limit_sec

        log.print(f"=== Begin: Gamma={GAMMA}, Alpha={ALPHA}, Beta={BETA} ===")

        exploration_eta = 2.0

        best_MIS_size_global = 0
        loss_history = []
        start_time = time.time()
        epoch = 0

        log.print("\n[Epoch Log Columns]")
        log.print("epoch, best_after_global_search, final_after_12swap, global_best, elapsed_sec")
        init_iter = 1
        while (time.time() - start_time) < TIME_LIMIT:
            log.print(f"\n--- Epoch {epoch} ---")

            # ============================ Phase 1: 256 parallel initial runs ============================
            input_tensor = init_parallel_population(d_base, n, device, epoch, exploration_eta)
            if init_iter ==1:
              topk_solutions = run_parallel_mgd_until_converge(
                input_tensor=input_tensor,
                adjacency_matrix_tensor=adjacency_matrix_tensor,
                device=device,
                GAMMA=GAMMA,
                BETA=BETA,
                ALPHA=ALPHA,
                ITERATION_T=ITERATION_T,
                top_k=TOP_K,
              )
              init_iter = 0

            if len(topk_solutions) > 0:
                log.print(f"  > Initial parallel phase found {len(topk_solutions)} solutions. Best size = {topk_solutions[0][0]}")
            else:
                log.print("  MGD did not converge in Phase 1.")

            # ============================ Phase 2: Global Reset with top-k parents ============================
      #      if len(topk_solutions) > 0:
      #          log.print(f"  > Starting {NUM_RESETS} global resets from random choice of top-{TOP_K}...")

       #         for r in range(NUM_RESETS):
            
            input_tensor = build_parallel_resets_from_topk(
                        topk_solutions=topk_solutions,
                        n=n,
                        device=device,
                        RESET_RATIO=RESET_RATIO,
                    )

            new_topk = run_parallel_mgd_until_converge(
                        input_tensor=input_tensor,
                        adjacency_matrix_tensor=adjacency_matrix_tensor,
                        device=device,
                        GAMMA=GAMMA,
                        BETA=BETA,
                        ALPHA=ALPHA,
                        ITERATION_T=ITERATION_T,
                        top_k=TOP_K,
                    )

            topk_solutions = merge_keep_topk(topk_solutions, new_topk, TOP_K)
            

            epoch_best_size = topk_solutions[0][0] if len(topk_solutions) > 0 else 0

            # ============================ Phase 3: Local Search on final top-k ============================
            final_epoch_size = 0
            final_epoch_nodes = set()

            if len(topk_solutions) > 0:
                log.print(f"  > Running local search on final top-{len(topk_solutions)} solutions...")

                for idx_k, (cand_size, cand_nodes) in enumerate(topk_solutions):
                    log.print(f"    Candidate {idx_k}: size before 1-2 swap = {cand_size}")

                    ls_size, ls_nodes = one_two_swap_adapter_from_adj(
                        adj_list_of_sets=adj_list_of_sets,
                        G_nx=G,
                        initial_mis_tensor=cand_nodes,
                        epoch_num=epoch,
                        log=log
                    )

                    M_check = list(ls_nodes)
                    feasible_final = strict_mis_feasibility_check(G, M_check)
                    if feasible_final:
                        log.print("    [Feasible FINAL MIS]: True")
                    else:
                        log.print("    [Feasible FINAL MIS]: False  (WARNING: infeasible set detected)")

                    if ls_size > final_epoch_size:
                        final_epoch_size = ls_size
                        final_epoch_nodes = set(ls_nodes)

            if final_epoch_size > best_MIS_size_global:
                best_MIS_size_global = final_epoch_size
                log.print(f"*** NEW GLOBAL BEST: {best_MIS_size_global} ***")

            elapsed = time.time() - start_time
            log.print(f"{epoch}, {epoch_best_size}, {final_epoch_size}, {best_MIS_size_global}, {elapsed:.4f}")

            epoch += 1

        runtime_sec = time.time() - start_time
        log.print("\n==========================================")
        log.print(f"Time's up! Best MIS found (this seed): {best_MIS_size_global}")
        log.print(f"Runtime (sec): {runtime_sec:.4f}")
        log.print(f"Run end: {now_str()}")
        log.print("==========================================")

        return best_MIS_size_global, runtime_sec

    finally:
        log.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_all_experiments():

    graph_settings = [
        # (3000, 0.03334),
        # # (8000, 0.7),
        # (3000, 0.10003),
        # (3000, 0.33344),
        # (1000, 0.10010),
        # (1000, 0.30030),
        # (1000, 0.50050),

        (10000, 0.5),
        (20000, 0.5),
        (30000, 0.5),
    ]

    seeds = list(range(8))
    time_limit_sec = 5 * 60
    # time_limit_sec = 30
    root_out = "./mis_logs_test-topk5"

    for (n, p) in graph_settings:
        beta, alpha = get_params_for_graph(n, p)
        out_dir = make_graph_dir(root_out, n, p, beta, alpha)

        best_list = []
        runtime_list = []

        for sd in seeds:
            best_sd, rt_sd = run_one_graph_one_seed(
                n=n,
                p=p,
                seed=sd,
                device=device,
                time_limit_sec=time_limit_sec,
                out_dir=out_dir,
            )
            best_list.append(best_sd)
            runtime_list.append(rt_sd)

        summary_path = os.path.join(out_dir, "summary.txt")
        avg_best = float(np.mean(best_list)) if len(best_list) > 0 else 0.0

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("==========================================\n")
            f.write(f"Summary: (n,p)=({n},{p})\n")
            f.write(f"Beta={beta}, Alpha={alpha}\n")
            f.write(f"Time limit per seed: {time_limit_sec} sec\n")
            f.write(f"Batch size={BATCH_SIZE}, Top-k={TOP_K}\n")
            f.write(f"Created: {now_str()}\n")
            f.write("==========================================\n\n")

            for i, sd in enumerate(seeds):
                f.write(f"seed={sd}: best={best_list[i]}, runtime_sec={runtime_list[i]:.4f}\n")

            f.write("\n------------------------------------------\n")
            f.write(f"Average best over {len(seeds)} seeds: {avg_best:.6f}\n")
            f.write("------------------------------------------\n")

        print(f"\n[DONE] (n,p)=({n},{p}) logs saved to: {out_dir}")
        print(f"       Summary: {summary_path}\n")

if __name__ == "__main__":
    run_all_experiments()
