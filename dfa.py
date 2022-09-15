#!/usr/bin/env python3

from IPython.display import display

from collections import defaultdict
from graphviz import Digraph
import numpy as np

class Dfa(object):
    def __init__(self, edges, n_symbols):
        self.edges = tuple(tuple(edge) for edge in edges)
        self.n_states = len(edges)
        self.n_symbols = n_symbols

        edge_symbols = [
            sym for state_edges in edges for sym, _ in state_edges
            if sym is not None
        ]
        if len(edge_symbols) == 0:
            self.n_symbols = 0
        else:
            self.n_symbols = max(edge_symbols) + 1

        self.n_neighbors = max(len(state_edges) for state_edges in edges)
        self.approx_ppl = np.mean([np.log(len(state_edges)) for state_edges in edges])
        self._compute_distances()
        self.unused_symbols = [i for i in range(n_symbols) if i not in edge_symbols]

    def sample(self, rand, start_state=0, stringify=False, max_len=32):
        state = start_state
        out = []
        for i in range(max_len):
            choices = self.edges[state]
            next_symbol, next_state = choices[rand.randint(len(choices))]
            if next_symbol is None:
                break
            out.append(next_symbol)
            state = next_state
        #if i == max_len - 1:
        #    print("warning: max len sampled")
        #    display(self.render())
        #    out = [chr(ord('a') + sym) for sym in out]
        #    print("".join(out))
        #    assert False
        if stringify:
            out = [chr(ord('a') + sym) for sym in out]
            return "".join(out)
        else:
            return tuple(out)

    def stationary_dist(self):
        if hasattr(self, "_cached_stationary_dist"):
            return self._cached_stationary_dist

        mat = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            neighbors = [nstate for _, nstate in self.edges[i] if nstate is not None]
            for neighbor in neighbors:
                mat[i, neighbor] += 1 / len(neighbors)
        for i in range(8):
            mat = mat @ mat
        self._cached_stationary_dist = mat[0, :]
        return self._cached_stationary_dist

    def predict_one(self, state):
        out = np.zeros(self.n_symbols + 1)
        for sym, _ in self.edges[state]:
            if sym is None:
                out[-1] += 1
            else:
                out[sym] += 1
        out /= out.sum()
        return out

    def prune(self):
        reachable = {0}
        terminable = {
            i for i in range(self.n_states) 
            if (None, None) in self.edges[i]
        }
        for i in range(self.n_states):
            new_reachable = set(reachable)
            new_terminable = set(terminable)
            for j in range(self.n_states):
                if j in reachable:
                    new_reachable |= {k for _, k in self.edges[j] if k is not None}
                if any(k in terminable for _, k in self.edges[j]):
                    new_terminable.add(j)
            reachable = new_reachable
            terminable = new_terminable
        new_edges = []
        for i in range(self.n_states):
            #if i not in reachable:
            #    new_edges.append([])
            #    continue
            #state_edges = [
            #    edge for edge in self.edges[i]
            #    if edge[1] is None or edge[1] in reachable
            #]
            #if i in reachable and i not in terminable:
            #    state_edges.append((None, None))
            #new_edges.append(state_edges)

            if not (i in reachable and i in terminable):
                state_edges = []
            else:
                state_edges = [
                    edge for edge in self.edges[i] 
                    if edge[1] is None or 
                        (edge[1] in terminable and edge[1] in reachable)
                ]
            new_edges.append(state_edges)
        return Dfa(new_edges, self.n_symbols)

    def is_empty(self):
        return self.n_symbols == 0  or all(state_edges == () for state_edges in self.edges)

    def render(self, compact=False):
        dot = Digraph()
        for i in range(len(self.edges)):
            if self.edges[i] == ():
                continue
            if (None, None) in self.edges[i]:
                dot.attr("node", shape="doublecircle")
            else:
                dot.attr("node", shape="circle")
            dot.node(str(i))
        for i in range(len(self.edges)):
            if compact:
                dsts = set(j for label, j in self.edges[i] if label is not None)
                for j in dsts:
                    dot.edge(str(i), str(j))
            else:
                for label, j in self.edges[i]:
                    if label is None:
                        continue
                    dot.edge(str(i), str(j), label=chr(ord('a') + label))
        return dot

    def adjacency_stats(self):
        out = []

        nstate_counts = []
        for edges in self.edges:
            distinct_nstates = set(nstate for _, nstate in edges if nstate is not None)
            nstate_counts.append(len(distinct_nstates))
        out.append(f"mean out-edges: {np.mean(nstate_counts):.2f}")

        symbol_counts = defaultdict(lambda: 0)
        for edges in self.edges:
            for sym, _ in edges:
                symbol_counts[sym] += 1

        score = sum(symbol_counts.values()) / (self.n_symbols - len(self.unused_symbols))
        out.append(f"mean in-edges: {score:.2f}")

        return "\n".join(out)


    def _compute_distances(self):
        distances = {
            state: {
                nstate: np.inf
                for nstate in range(self.n_states)
            } for state in range(self.n_states)
        }
        for state in range(self.n_states):
            for _, nstate in self.edges[state]:
                distances[state][nstate] = 1
        for k in range(self.n_states):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    distances[i][j] = min(
                        distances[i][j],
                        distances[i][k] + distances[k][j]
                    )
        self.distances = distances

    def annotate(self, seq, initial_state=0, allow_incomplete=True):
        state = initial_state
        states = [state]
        for symbol in seq:
            state_edges = self.edges[state]
            next_state, = [
                nstate for nsym, nstate in state_edges if nsym == symbol
            ]
            states.append(next_state)
            state = next_state
        if not allow_incomplete:
            assert (None, None) in self.edges[state]
        return states

    @classmethod
    def generate_balanced(cls, rand, n_states, n_symbols, n_adjacencies):
        while True:
            symbol_counts = {
                symbol: n_adjacencies
                for symbol in range(n_symbols)
            }
            edges = []
            for i in range(n_states):
                #out_symbols = rand.choice(available_symbols, size=n_adjacencies, replace=False)
                neighbors = rand.choice(n_states, size=n_adjacencies, replace=False)
                state_edges = []
                used_symbols = set() 
                for neighbor in neighbors:
                    #for _ in range(n_symbols // n_states):
                    neighbor_edges = []
                    for _ in range(n_symbols):
                        available_symbols = [s for s, c in symbol_counts.items() if c > 0]
                        if len(available_symbols) == 0:
                            break
                        symbol = rand.choice(available_symbols)
                        if symbol in used_symbols:
                            continue
                        state_edges.append((symbol, neighbor))
                        neighbor_edges.append((symbol, neighbor))
                        used_symbols.add(symbol)
                        symbol_counts[symbol] -= 1
                        if len(neighbor_edges) == n_symbols // n_states:
                            break

            #n_symbols * n_adjacencies
            #n_states * n_adjacencies * n_symbols


                #state_edges = []
                #for symbol in out_symbols:
                #    neighbor = rand.randint(n_states)
                #    state_edges.append((symbol, neighbor))
                #    symbol_counts[symbol] -= 1

                if rand.random() < 0.5:
                    state_edges.append((None, None))
                edges.append(state_edges)

            dfa = Dfa(edges, n_symbols).prune()
            if dfa.is_empty():
                continue
            return dfa

    @classmethod
    def generate(cls, rand, n_states, n_symbols, n_neighbors):
        while True:
            #symbols_for_state = {i: set() for i in range(n_states)}
            #for symbol in range(n_symbols):
            #    n_states_for_symbol = rand.geometric(0.5)
            #    states_for_symbol = rand.choice(n_states, size=n_states_for_symbol, replacement=False)
            #    for state in states_for_symbol:
            #        symbols_for_state[state].add(symbol)

            edges = []
            for i in range(n_states):
                state_edges = []
                available_symbols = list(range(n_symbols))
                for j in range(n_neighbors):
                    sym = rand.choice(available_symbols)
                    neighbor = rand.randint(n_states)
                    state_edges.append((sym, neighbor))
                    available_symbols.remove(sym)

                if rand.random() < 0.5:
                    state_edges.append((None, None))

                edges.append(state_edges)

                #neighbors = rand.choice(n_states, size=n_states//2, replacement=True)
                #neighbors = sorted(set(neigbors))
                #for neighbor in neighbors:
                #    neighbor_symbols = symbols_for_state[neighbor]

            dfa = Dfa(edges).prune()
            if dfa.is_empty():
                continue
            dfa._compute_distances()
            return dfa
