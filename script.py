from collections import defaultdict
from dfa import Dfa
from model import SequenceModel, Ngram, GaussianProbe
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch import nn, optim
import torch
import json

N_STATES = 8
N_SYMBOLS = 128
N_ADJACENCIES = 4

#N_SYMBOLS = 8
#N_ADJACENCIES = 4

N_ITERS = 2000

def train_model(dfa, rand, train, model_type, **train_params):
    return SequenceModel.train_model(dfa, rand, 256, 256, max_iters=(N_ITERS if train else 0), model_type=model_type, **train_params)

def train_ngram_model(dfa, rand, max_order):
    return Ngram.train_model(dfa, rand, max_order, n_samples=128 * 1000 * 2)

def compute_interpolator(contexts, true_label, pred_labels, add=True):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.ParameterList([nn.Parameter(torch.tensor(0).float().cuda()) for _ in pred_labels])
            assert len(pred_labels) == 2
            #self.param = nn.Parameter(torch.tensor(0).float().cuda())
            self.loss = nn.MSELoss(reduction="none")
        
        def forward(self, true_dist, pred_dists):
            if add:
                param_sum = sum([torch.exp(p) for p in self.params])
                params_norm = [torch.exp(p) / param_sum for p in self.params]
                #param = torch.sigmoid(self.param)
                #oparam = 1 - param
            else:
                #true_dist = torch.log(true_dist + 1e-6)
                pred_dists = [torch.log(dist + 1e-6) for dist in pred_dists]
                params_norm = [torch.exp(p) for p in self.params]
                #param = self.param
                #oparam = -param
            pred = [param * dist for param, dist in zip(params_norm, pred_dists)]
            pred = sum(pred)
            #if not add:
            #    print(param, pred_dists[0][:10])
            #    print(oparam, pred_dists[1][:10])
            #pred = param * pred_dists[0] + oparam * pred_dists[1]
            if not add:
                #print(pred.shape)
                #assert False
                pred = pred.softmax(dim=1)
                assert (pred == pred).all(), (params_norm, pred_dists[0][0, :10], pred_dists[1][0, :10], pred[0, :10])
            #print(pred_dists)
            #print(pred)
            #print(true_dist)
            loss = self.loss(pred, true_dist).sum(dim=1).mean(dim=0)
            err = torch.abs(pred - true_dist).sum(dim=1).mean(dim=0) / 2
            assert err <= 1, (pred[0, :10], true_dist[0, :10])
            #if not add:
            #    print(pred[0][:10])
            #    print(loss, err)
            return loss, err
        
    model = Model()
    opt = optim.Adam(model.parameters(), lr=0.01)
    true_data = torch.tensor([contexts[k][true_label] for k in contexts]).float().cuda()
    pred_data = [torch.tensor([contexts[k][pred_label] for k in contexts]).float().cuda() for pred_label in pred_labels]
    for i in range(200):
        loss, err = model(true_data, pred_data)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    #if not add:
    #    assert False
    
    return err.item(), [p.item() for p in model.params]

def run_experiment_1(dfa, data_rand, hypotheses, model_type, **train_params):
    rnn = train_model(dfa, data_rand, train=True, model_type=model_type, **train_params)
    rnn_copy = train_model(dfa, data_rand, train=True, model_type=model_type, **train_params)
    ngram = None
    if any("gram" in name for name, _ in hypotheses):
        ngram = train_ngram_model(dfa, data_rand, max_order=2)
    
    ghost_symbols = []
    real_symbols = []
    while len(ghost_symbols) < 100:
        prefix = dfa.sample(data_rand)
        if len(prefix) == 0:
            continue
        prefix_len = data_rand.randint(len(prefix))
        prefix_symbols = prefix[:prefix_len]
        ann = dfa.annotate(prefix_symbols)
        prefix_state = ann[-1]
        edges = dfa.edges[prefix_state]
        
        forbidden_symbols = [sym for sym, dst in edges] + dfa.unused_symbols
        ghost_choices = [i for i in range(dfa.n_symbols) if i not in forbidden_symbols]
        if len(ghost_choices) == 0:
            continue
        ghost_symbol = data_rand.choice(ghost_choices)
        ghost_symbols.append(prefix_symbols + (ghost_symbol,))
        
        real_choices = [sym for sym, dst in edges if sym is not None]
        real_symbol = data_rand.choice(real_choices)
        real_symbols.append(prefix_symbols + (real_symbol,))
    
    results = []
    ghost_contexts = defaultdict(dict)
    real_contexts = defaultdict(dict)
    for hypothesis_name, hypothesis in hypotheses:
        total_interp_dist = 0
        total_extrap_dist = 0
        count = 0
        for i in range(len(ghost_symbols)):
            
            if False:
                rs = real_symbols[i]
                true_states = dfa.annotate(rs[:-1], allow_incomplete=True)
                if rs in real_contexts:
                    true_preds = real_contexts[rs]["TRUE"]
                    copy_preds = real_contexts[rs]["COPY"]
                    lookahead_preds = real_contexts[rs]["LOOKAHEAD"]
                else:
                    true_preds, = rnn.predict_one([rs])
                    copy_preds, = rnn_copy.predict_one([rs])
                    #lookahead_preds = rnn.predict_two([rs], 2 * N_ADJACENCIES, data_rand)
                    lookahead_preds = None
                    real_contexts[rs]["TRUE"] = true_preds
                    real_contexts[rs]["COPY"] = copy_preds
                    real_contexts[rs]["LOOKAHEAD"] = lookahead_preds
                real_preds = hypothesis(
                    dfa, rs, true_states,
                    model_preds=true_preds, copy_preds=copy_preds, ngram_model=ngram,
                    lookahead_preds=lookahead_preds,
                    **train_params
                )
                real_contexts[rs][hypothesis_name.strip()] = real_preds
                total_interp_dist += np.abs(true_preds - real_preds).sum() / 2
            
            # -----
            
            gs = ghost_symbols[i]
            true_states = dfa.annotate(gs[:-1], allow_incomplete=True)
            if gs in ghost_contexts:
                true_preds = ghost_contexts[gs]["TRUE"]
                copy_preds = ghost_contexts[gs]["COPY"]
                lookahead_preds = ghost_contexts[gs]["LOOKAHEAD"]
            else:
                true_preds, = rnn.predict_one([gs])
                copy_preds, = rnn_copy.predict_one([gs])
                #lookahead_preds = rnn.predict_two([gs], 2 * N_ADJACENCIES, data_rand)
                lookahead_preds = None
                ghost_contexts[gs]["TRUE"] = true_preds
                ghost_contexts[gs]["COPY"] = copy_preds
                ghost_contexts[gs]["LOOKAHEAD"] = lookahead_preds

            ghost_preds = hypothesis(
                dfa, gs, true_states, 
                model_preds=true_preds, copy_preds=copy_preds, ngram_model=ngram,
                lookahead_preds=lookahead_preds,
                **train_params
            )
            ghost_contexts[gs][hypothesis_name.strip()] = ghost_preds
            total_extrap_dist += np.abs(true_preds - ghost_preds).sum() / 2
            
            # -----
            
            count += 1
            
        results.append({
            "n_states": int(dfa.n_states),
            "n_symbols": int(dfa.n_symbols),
            "n_neighbors": int(dfa.n_neighbors),
            "hypothesis": hypothesis_name.strip(),
            "interp_acc": float(total_interp_dist / count),
            "extrap_acc": float(total_extrap_dist / count),
            **{k: float(v) for k, v in train_params.items()}
        })
        
    #interp_interpolated_acc, interp_interpolation_weights = compute_interpolator(real_contexts, "TRUE", ["ngram(2)", "mean_dist_state"])
    extrap_interpolated_acc_add, extrap_interpolation_weights_add = compute_interpolator(ghost_contexts, "TRUE", ["ngram(2)", "mean_dist_state"], add=True)
    extrap_interpolated_acc_mul, extrap_interpolation_weights_mul = compute_interpolator(ghost_contexts, "TRUE", ["ngram(2)", "mean_dist_state"], add=False)
    results.append({
        "n_states": int(dfa.n_states),
        "n_symbols": int(dfa.n_symbols),
        "n_neighbors": int(dfa.n_neighbors),
        "hypothesis": "interp_add",
        #"interp_acc": interp_interpolated_acc,
        "extrap_acc": float(extrap_interpolated_acc_add),
        #"interp_weight_ngram": interp_interpolation_weights[0],
        #"interp_weight_out": interp_interpolation_weights[1],
        "extrap_weight_ngram": float(extrap_interpolation_weights_add[0]),
        "extrap_weight_out": float(extrap_interpolation_weights_add[1]),
        **{k: float(v) for k, v in train_params.items()}
    })
    
    results.append({
        "n_states": int(dfa.n_states),
        "n_symbols": int(dfa.n_symbols),
        "n_neighbors": int(dfa.n_neighbors),
        "hypothesis": "interp_mul",
        #"interp_acc": interp_interpolated_acc,
        "extrap_acc": float(extrap_interpolated_acc_mul),
        #"interp_weight_ngram": interp_interpolation_weights[0],
        #"interp_weight_out": interp_interpolation_weights[1],
        "extrap_weight_ngram": float(extrap_interpolation_weights_mul[0]),
        "extrap_weight_out": float(extrap_interpolation_weights_mul[1]),
        **{k: float(v) for k, v in train_params.items()}
    })
    
    return results

hyp_rand = np.random.RandomState(0)

def random_out_state_hyp(dfa, symbols, true_states, **kwargs):
    state_edges = dfa.edges[true_states[-1]]
    state_edges = [edge for edge in state_edges if edge != (None, None)]
    reachable = sorted(set(state for _, state in state_edges))
    predicted_state = hyp_rand.choice(reachable)
    return dfa.predict_one(predicted_state)
    
def random_in_state_hyp(dfa, symbols, true_states, **kwargs):
    reachable = [nstate for state_edges in dfa.edges for symbol, nstate in state_edges if symbol == symbols[-1]]
    reachable = sorted(set(reachable))
    predicted_state = hyp_rand.choice(reachable)
    return dfa.predict_one(predicted_state)

def mean_out_state_hyp(dfa, symbols, true_states, **kwargs):
    state_edges = dfa.edges[true_states[-1]]
    state_edges = [edge for edge in state_edges if edge != (None, None)]
    reachable = sorted(set(state for _, state in state_edges))
    dists = [dfa.predict_one(state) for state in reachable]
    #print(reachable)
    return np.mean(dists, axis=0)

def mean_dist_state_hyp(dfa, symbols, true_states, **kwargs):
    dists = 0
    weights = 0
    for state in range(dfa.n_states):
        weight = np.exp(-dfa.distances[true_states[-1]][state])
        #print(weight)
        dists += (weight * dfa.predict_one(state))
        weights += weight
    dist = dists / weights
    return dist

def mean_in_state_hyp(dfa, symbols, true_states, **kwargs):
    reachable = [nstate for state_edges in dfa.edges for symbol, nstate in state_edges if symbol == symbols[-1]]
    reachable = sorted(set(reachable))
    dists = [dfa.predict_one(state) for state in reachable]
    #print(reachable)
    return np.mean(dists, axis=0)

def oracle_out_state_hyp(dfa, symbols, true_states, model_preds, **kwargs):
    state_edges = dfa.edges[true_states[-1]]
    state_edges = [edge for edge in state_edges if edge != (None, None)]
    reachable = sorted(set(state for _, state in state_edges))
    dists = [dfa.predict_one(state) for state in reachable]
    return min(dists, key=lambda x: np.abs(x - model_preds).sum())

def oracle_in_state_hyp(dfa, symbols, true_states, model_preds, **kwargs):
    reachable = [nstate for state_edges in dfa.edges for symbol, nstate in state_edges if symbol == symbols[-1]]
    reachable = sorted(set(reachable))
    dists = [dfa.predict_one(state) for state in reachable]
    return min(dists, key=lambda x: np.abs(x - model_preds).sum())

def copycat_hyp(dfa, symbols, true_states, copy_preds, **kwargs):
    return copy_preds

def lookahead_hyp(dfa, symbols, true_states, lookahead_preds, **kwargs):
    return sum(lookahead_preds) / len(lookahead_preds)

def skip_hyp(dfa, symbols, true_states, **kwargs):
    return dfa.predict_one(true_states[-1])

def gt_hyp(dfa, symbols, true_states, **kwargs):
    try:
        state = dfa.annotate(symbols)[-1]
        #print("gt state", state)
        return dfa.predict_one(state)
    except:
        return unif_hyp(dfa, symbols, true_states, **kwargs)

def ngram_hyp(order):
    assert order > 0
    def fn(dfa, symbols, true_states, ngram_model, **kwargs):
        if order == 1:
            dist = ngram_model.predict_symbol(())
        else:
            dist = ngram_model.predict_symbol(symbols[-order+1:])
        out = np.zeros(N_SYMBOLS+1)
        for i in range(N_SYMBOLS):
            if i in dist:
                out[i] = dist[i]
        
        if None in dist:
            out[-1] = dist[None]
            
       # print("orig", out.shape)
        
        #display(dfa.render())
        #print([chr(ord("a") + s) for s in symbols])
        #print(model_preds)
        #print(out)
        #print(np.abs(model_preds - out))
        #assert False
        return out
    return fn

def ngram_state_hyp(order):
    assert order > 0
    def fn(dfa, symbols, true_states, ngram_model, **kwargs):
        states = ngram_model.predict_state(symbols[-order:])
        #print("inferred states", states)
        out = np.zeros(N_SYMBOLS+1)
        for state in states:
            out += dfa.predict_one(state) * states[state]
        assert abs(out.sum() - 1) < 1e-4
        
        direct = ngram_hyp(order)(dfa, symbols, true_states, ngram_model)
        #print(out)
        #print(direct)
        #print([chr(ord("a") + s) for s in symbols])
        
        #print(out)
        #print(gt_hyp(dfa, symbols, true_states, **kwargs))
        #print()
        #print(out.shape)
        return out
    return fn

def mean_both_state_hyp(dfa, symbols, true_states, ngram_model, **kwargs):
    #return (mean_in_state_hyp(dfa, symbols, true_states) + mean_out_state_hyp(dfa, symbols, true_states)) / 2
    return (
        ngram_state_hyp(2)(dfa, symbols, true_states, ngram_model)
        + mean_out_state_hyp(dfa, symbols, true_states)
    ) / 2

def unif_hyp(dfa, symbols, true_states, **kwargs):
    return np.ones(dfa.n_symbols + 1, dtype=np.float32) / (dfa.n_symbols + 1)

for model_type in ["lstm", "transformer"]:
    results1 = []
    configs = [
#{"symbol_swap": [0, 1/2]},
{"symbol_swap": [0, 1/4, 1/2, 3/4]},
{"symbol_mask": [0, 1/4, 1/2, 3/4]},
{"symbol_dropout": [0, 1/4, 1/2, 3/4]},
{"state_reset": [0, 1/4, 1/2, 3/4]},
{"state_skip": [0, 1/4, 1/2, 3/4]},
{"state_noise": [0, 1/2, 1, 2] if model_type == "lstm" else [0]},
{"state_dropout": [0, 1/4, 1/2, 3/4]}
    ]

    for config in configs:
        assert len(config) == 1
        noise_type, = config.keys()
        noise_values = config[noise_type]
        result_group = []
        for noise_value in noise_values:
            dfa_rand = np.random.RandomState(0)
            for i in tqdm(range(3)):
               dfa = Dfa.generate_balanced(dfa_rand, N_STATES, N_SYMBOLS, N_ADJACENCIES)
               hypotheses = [
                   ("copycat          ", copycat_hyp),
                   ("gt               ", gt_hyp),
                   #("ngram(2)x         ", ngram_hyp(4)),
                   #("ngram(4)   ", ngram_state_hyp(4)),
                   ("ngram(2)   ", ngram_hyp(2)),
                   #("ngram_state(2)", ngram_state_hyp(2)),
                   ("ngram(1)   ", ngram_hyp(1)),
                   ("skip", skip_hyp),
                   #("lookahead        ", lookahead_hyp),
                   ("mean_dist_state  ", mean_dist_state_hyp),
               ]
               #display(dfa.render())
               data_rand = np.random.RandomState(0)
               result_group += run_experiment_1(
                   dfa,
                   data_rand,
                   hypotheses,
                               model_type,
                   **{noise_type: noise_value}
               )
               #assert False
    
        results1.append(result_group)
        result_group = pd.DataFrame(result_group)
    
    #sns.relplot(data=result_group, hue="hypothesis", x=noise_type, y="interp_acc", aspect=1, kind="line")
    #plt.title(noise_type)
    #plt.ylim(0, 0.5)
    #plt.show()
    
    #sns.relplot(data=result_group, hue="hypothesis", x=noise_type, y="extrap_acc", aspect=1, kind="line")
    #plt.title(noise_type)
    #plt.ylim(0, 0.5)
    #plt.show()
    
    #sns.relplot(
    #    data=result_group.melt(id_vars=[noise_type], value_vars=["interp_weight_ngram", "interp_weight_out"]),
    #    x=noise_type, y="value", hue="variable", aspect=1, kind="line"
    #)
    #plt.show()
    
    #sns.relplot(
    #    data=result_group.melt(id_vars=[noise_type], value_vars=["extrap_weight_ngram", "extrap_weight_out"]),
    #    x=noise_type, y="value", hue="variable", aspect=1, kind="line"
    #    #data=result_group, x=noise_type, y="extrap_weight", hue="hypothesis", aspect=1, kind="line"
    #)
    #plt.show()
    
    with open(model_type + ".json", "w") as writer:
        json.dump(results1, writer)
