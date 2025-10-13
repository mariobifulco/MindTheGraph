from math import ceil, sqrt
import os
from itertools import product
from statistics import fmean, median, mode
from time import strftime
from typing import List, Tuple, Dict, Any

from minorminer import find_embedding as embed
from pandas import DataFrame, read_csv

import dwave_networkx as dnx
import networkx as nx
import wandb

SEED = 5219962669
WANDB = False

conf_done = set()
save_path = f"embedding_{strftime("%Y%m%d_%H%M%S")}.csv"


def init_conf_done():
    global conf_done
    filename = "embedding.csv"

    if not os.path.exists(filename):
        return

    df = read_csv(filename)
    conf_done = set(map(tuple, df[["qpu_type", "arg1", "arg2"]].to_numpy()))


def max_clique_for_qpu(qpu_graph: nx.Graph) -> Tuple[int, Any]:
    low, high = 2, ceil(sqrt(qpu_graph.number_of_nodes()))
    best_dim = 1
    best_embedding = None
    while low <= high:
        mid = (low + high) // 2
        problem_graph = nx.complete_graph(mid)
        try:
            embedding, is_valid = embed(
                S=problem_graph, T=qpu_graph, random_seed=SEED, return_overlap=True, threads=4, tries=5)
        except RuntimeError:
            is_valid, embedding = False, None
        if is_valid:
            best_dim = mid
            best_embedding = embedding
            low = mid + 1
        else:
            high = mid - 1
    return best_dim, best_embedding


def generate_confs() -> List[Tuple[str, int, int]]:
    zephyrs = [("zephyr", m, t) for m, t in product(range(2, 8), range(1, 26))]
    havel_hakimis = [("havel_hakimi", d, q)
                     for d, q in product(range(5, 106, 25), range(50, 10500, 350))]
    qpu_confs = []
    for i in range(len(zephyrs)):
        qpu_confs.append(zephyrs[i])
        qpu_confs.append(havel_hakimis[i])
    return qpu_confs


def dispatch_qpu(qtype: str, arg1: int, arg2: int) -> nx.Graph:
    if qtype == "zephyr":
        # m == arg1, t == arg2
        return dnx.zephyr_graph(arg1, arg2)
    elif qtype == "havel_hakimi":
        # deg == arg1, num_qubits == arg2
        return nx.havel_hakimi_graph([arg1] * arg2)
    else:
        assert False, "unreachable"


def save_data(data: List[Dict]):
    df = DataFrame(data)
    df.to_csv(save_path, index=False)
    if WANDB:
        artifact = wandb.Artifact("embedding_results", type="dataset")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
    print(f"Saved to {save_path}")


def single_conf_run(qtype: str, arg1: int, arg2: int) -> Dict:
    qpu = dispatch_qpu(qtype, arg1, arg2)
    print(qpu.number_of_nodes(), qpu.number_of_edges(),
          sum(dict(qpu.degree()).values()) / len(qpu))
    clique_dim, embedding = max_clique_for_qpu(qpu)
    chain_lens = [len(x) for x in embedding.values()]
    return {"qpu_type": qtype, "arg1": arg1, "arg2": arg2, "nodes": qpu.number_of_nodes(),
            "edges": qpu.number_of_edges(), "avg_deg": sum(dict(qpu.degree()).values()) / len(qpu),
            "clique_dim": clique_dim, "avg_chain_len": fmean(chain_lens), "median_chain_len": median(chain_lens),
            "mode_chain_len": mode(chain_lens)}


def main():
    if WANDB:
        wandb.init(project="MtG", config={"seed": SEED})
    init_conf_done()
    res = []
    qpu_confs = generate_confs()
    tot = len(qpu_confs)

    for i, (qtype, arg1, arg2) in enumerate(qpu_confs):
        if (qtype, arg1, arg2) in conf_done:
            print(f"{i + 1}/{tot}) {qtype},{arg1},{arg2} ALREADY DONE, SKIPPING")
            continue
        try:
            status = single_conf_run(qtype, arg1, arg2)
            res.append(status)
            print(f"{i + 1}/{tot}) {qtype},{arg1},{arg2}")
        except Exception as e:
            print(f"{i + 1}/{tot}) {qtype},{arg1},{arg2} FAILED: {e}")
        if i != 0 and i % 5 == 0:
            save_data(res)
    save_data(res)


if __name__ == "__main__":
    main()
