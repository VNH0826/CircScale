#!/usr/bin/env python3
import os
import argparse
import re
import torch
from pathlib import Path
from collections import defaultdict, deque
from torch_geometric.data import Data


def parse_bench_file(bench_file: Path):
    nodes = {}
    inputs = []
    outputs = []
    with open(bench_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('INPUT('):
                name = line.split('(')[1].split(')')[0]
                inputs.append(name)
                nodes.setdefault(name, {'type': 0, 'inputs': []})
            elif line.startswith('OUTPUT('):
                name = line.split('(')[1].split(')')[0]
                outputs.append(name)
            elif ' = AND(' in line:
                left, rest = line.split(' = AND(')
                node = left.strip()
                args = rest.split(')')[0].split(',')
                ins = [a.strip() for a in args if a.strip()]
                nodes[node] = {'type': 2, 'inputs': ins, 'is_and': True}
            elif ' = NOT(' in line:
                left, rest = line.split(' = NOT(')
                node = left.strip()
                src = rest.split(')')[0].strip()
                nodes[node] = {'type': 2, 'inputs': [src], 'is_not': True}
    return nodes, inputs, outputs


def build_graph(nodes: dict, inputs: list, outputs: list):
    # node_types: 0=PI, 1=PO, 2=GATE
    node_types = {}
    all_nodes = set(nodes.keys()) | set(inputs) | set(outputs)

    # Ensure all outputs are present as nodes (if missing, create empty gate with no inputs)
    for o in outputs:
        all_nodes.add(o)
        nodes.setdefault(o, {'type': 2, 'inputs': []})

    for n in all_nodes:
        if n in inputs:
            node_types[n] = 0
        elif n in outputs:
            node_types[n] = 1
        else:
            node_types[n] = 2

    # Build adjacency and compute inverted_count for AND nodes
    out_adj = defaultdict(list)
    indeg = defaultdict(int)
    inverted_from_not = set()

    for dst, data in nodes.items():
        for s in data.get('inputs', []):
            out_adj[s].append(dst)
            indeg[dst] += 1
            indeg.setdefault(s, 0)
        if data.get('is_not'):
            # Edges leaving NOT nodes are logically inverted
            inverted_from_not.add(dst)

    # Depth via topo (AIG semantics: NOT has 0-cost inversion)
    # Only transitions through non-NOT nodes increase depth by 1
    depth = {n: 0 for n in all_nodes if indeg.get(n, 0) == 0}
    q = deque([n for n in all_nodes if indeg.get(n, 0) == 0])
    while q:
        u = q.popleft()
        for v in out_adj.get(u, []):
            inc = 0 if nodes.get(u, {}).get('is_not') else 1
            depth[v] = max(depth.get(v, 0), depth.get(u, 0) + inc)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    for n in all_nodes:
        depth.setdefault(n, 0)

    # Inverted input count for gates: count predecessors that are NOT nodes
    def count_inverted(n: str) -> int:
        data = nodes.get(n, {})
        cnt = 0
        for s in data.get('inputs', []):
            if nodes.get(s, {}).get('is_not'):
                cnt += 1
        return cnt

    node_list = list(all_nodes)
    idx = {n: i for i, n in enumerate(node_list)}

    # Features x: [type, inverted_inputs]
    x = []
    for n in node_list:
        t = float(node_types[n])
        inv = float(count_inverted(n)) if t == 2 else 0.0
        x.append([t, inv])
    x = torch.tensor(x, dtype=torch.float)

    # Edges
    edges = []
    for dst, data in nodes.items():
        di = idx.get(dst)
        if di is None:
            continue
        for s in data.get('inputs', []):
            si = idx.get(s)
            if si is not None:
                edges.append([si, di])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    node_depth = torch.tensor([depth[n] for n in node_list], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, node_depth=node_depth)


def convert_dir(input_dir: Path, output_dir: Path, verbose: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_files = sorted([p for p in input_dir.glob('*.bench')])
    if not bench_files:
        print(f"No .bench files in {input_dir}")
        return
    ok = 0
    for f in bench_files:
        try:
            nodes, inputs, outputs = parse_bench_file(f)
            data = build_graph(nodes, inputs, outputs)
            torch.save(data, output_dir / f.with_suffix('.pt').name)
            ok += 1
            if verbose:
                print(f"{f.name}: nodes={data.x.size(0)} edges={data.edge_index.size(1)} maxD={int(data.node_depth.max())}")
        except Exception as e:
            print(f"[ERR] {f.name}: {e}")
    print(f"Saved {ok}/{len(bench_files)} .pt files to {output_dir}")


def main():
    ap = argparse.ArgumentParser(description='Convert AND/NOT bench to PyG .pt graphs')
    ap.add_argument('--input', required=True, help='Input bench file or directory')
    ap.add_argument('--output', required=True, help='Output directory')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    if in_path.is_dir():
        convert_dir(in_path, out_dir, args.verbose)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        nodes, inputs, outputs = parse_bench_file(in_path)
        data = build_graph(nodes, inputs, outputs)
        torch.save(data, out_dir / in_path.with_suffix('.pt').name)
        if args.verbose:
            print(f"{in_path.name}: nodes={data.x.size(0)} edges={data.edge_index.size(1)} maxD={int(data.node_depth.max())}")
        print(f"Saved 1/1 .pt files to {out_dir}")


if __name__ == '__main__':
    main() 