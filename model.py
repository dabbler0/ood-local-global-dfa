#!/usr/bin/env python3

from collections import defaultdict, Counter
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class Ngram():
    def __init__(self, n_symbols, context_to_state, context_to_symbol):
        self.sos = n_symbols
        self.context_to_state = context_to_state
        self.context_to_symbol = context_to_symbol

    def predict_state(self, context):
        #best = None
        context = (self.sos,) + context
        for n in reversed(range(len(context)+1)):
            if n == 0:
                ncontext = ()
            else:
                ncontext = tuple(context[-n:])
            if ncontext in self.context_to_state:
                probs = self.context_to_state[ncontext]
                return probs
                #if best is None:
                #    best = max(probs, key=lambda state: probs[state])
        return best
        #return max(counts, key=lambda state: counts[state])

        #print(context)
        #for n in reversed(range(len(context)+1)):
        #    ncontext = tuple(context[-n:])
        #    print(ncontext)
        #print(self.contexts.keys())
        #assert False

    def predict_symbol(self, context):
        context = (self.sos,) + context
        #print(context)
        #print(self.context_to_symbol)
        for n in reversed(range(len(context)+1)):
            if n == 0:
                ncontext = ()
            else:
                ncontext = tuple(context[-n:])
            #print(ncontext)
            if ncontext in self.context_to_symbol:
                probs = self.context_to_symbol[ncontext]
                return probs

        assert False

    @classmethod
    def train_model(cls, dfa, data_rand, max_order, n_samples):
        state_counter = defaultdict(Counter)
        symbol_counter = defaultdict(Counter)
        for t in range(n_samples):
            seq = dfa.sample(data_rand)
            ann = dfa.annotate(seq)
            seq = (dfa.n_symbols,) + seq + (None,)
            for n in range(max_order+1):
                for i in range(len(seq) - n + 1):
                    ctx = tuple(seq[i:i+n])
                    if i < len(seq) - n:
                        state = ann[i+n-1]
                        state_counter[ctx][state] += 1
                    if len(ctx) > 0:
                        symbol_counter[ctx[:-1]][ctx[-1]] += 1

        context_to_state = {}
        context_to_symbol = {}
        for ctx, counts in state_counter.items():
            denom = sum(counts.values())
            context_to_state[ctx] = {
                state: counts[state] / denom
                for state in counts
            }

        for ctx, counts in symbol_counter.items():
            denom = sum(counts.values())
            context_to_symbol[ctx] = {
                sym: counts[sym] / denom
                for sym in counts
            }

        return Ngram(dfa.n_symbols, context_to_state, context_to_symbol)


class SequenceModel(nn.Module):
    def __init__(
            self, n_states, n_symbols, n_embed, n_hidden, model_type,
            symbol_dropout, state_dropout
    ):
        super().__init__()
        n_symbols = int(n_symbols)
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.n_vocab = n_symbols + 3
        self.n_embed = n_embed
        self.n_hidden = n_hidden

        self.symbol_dropout = symbol_dropout
        self.state_dropout = state_dropout

        self.sos = n_symbols
        self.eos = n_symbols + 1
        self.pad = n_symbols + 2

        self.embed = nn.Sequential(
            nn.Embedding(self.n_vocab, n_embed),
            nn.Dropout(symbol_dropout),
        )
        self.state_dropout = nn.Dropout(state_dropout)

        if model_type == "rnn":
            self.rnn = nn.RNN(n_embed, n_hidden)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(n_embed, n_hidden)
        elif model_type == "gru":
            self.rnn = nn.GRU(n_embed, n_hidden)
        elif model_type == "transformer":
            assert n_embed == n_hidden
            layer = CustomTransformerEncoderLayer(n_hidden, 4, n_hidden, 0,
                    "relu", head_dropout=state_dropout)
            layer_norm = nn.LayerNorm(n_hidden)
            self.transformer = nn.TransformerEncoder(
                layer, 4, layer_norm
            )
            self.positional = PositionalEncoding(n_hidden, dropout=0)
            #self._reset_parameters()
        else:
            assert False
        self.pred = nn.Linear(n_hidden, self.n_vocab)

    def forward(self, seqs, state_reset=0, state_skip=0, state_noise=0):
        if hasattr(self, "rnn"):
            return self._forward_rnn(seqs, state_reset, state_skip, state_noise)
        elif hasattr(self, "transformer"):
            return self._forward_transformer(seqs, state_reset, state_skip, state_noise)
        else:
            assert False

    def _forward_rnn(self, seqs, state_reset, state_skip, state_noise):
        embedded = self.embed(seqs)
        if state_noise == 0 and state_reset == 0 and state_skip == 0:
            encoded, _ = self.rnn(embedded)
        else:
            encoded, _ = self._unroll_rnn(embedded, state_reset, state_skip, state_noise)
        preds = self.pred(encoded)
        return preds, encoded

    def _forward_transformer(self, seqs, state_reset, state_skip, state_noise):
        sz = seqs.shape[0]
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        assert state_noise == 0
        for i in range(sz):
            rand = torch.rand(1).item()
            if rand < state_reset:
                mask[i:, :i] = 0
            elif rand < state_skip:
                mask[:, i] = 0
                mask[i, i] = 1

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        embedded = self.embed(seqs)
        embedded = self.positional(embedded)
        encoded = self.transformer(embedded, mask.to(embedded.device))
        preds = self.pred(encoded)
        return preds, encoded

    def _unroll_rnn(self, embedded, state_reset, state_skip, state_noise):
        assert self.training
        def reset_state():
            hidden = [
                torch.zeros(1, embedded.shape[1], self.n_hidden).cuda(),
                torch.zeros(1, embedded.shape[1], self.n_hidden).cuda(),
            ]
            hidden = [
                h + torch.randn(h.shape).cuda() * state_noise
                for h in hidden
            ]
            return hidden

        hidden = reset_state()
        hidden = [self.state_dropout(h) for h in hidden]
        outs = []
        for i in range(embedded.shape[0]):
            old_hidden = hidden
            out, hidden = self.rnn(embedded[i:i+1, :, :], hidden)
            hidden = [
                h + torch.randn(h.shape).cuda() * state_noise
                for h in hidden
            ]
            rand = torch.rand(1).item()
            if rand < state_reset:
                hidden = reset_state()
            hidden = [self.state_dropout(h) for h in hidden]
            outs.append(out)
            if rand < state_skip:
                hidden = old_hidden
        return torch.cat(outs, dim=0), hidden

    def predict_one(self, symbols, eval_indices=None):
        assert not self.training
        seqs = [
            torch.tensor((self.sos,) + seq) for seq in symbols
        ]
        batch = pad_sequence(seqs, padding_value=self.pad).cuda()
        with torch.no_grad():
            preds, _ = self(batch)
        dists = []
        for i, seq in enumerate(symbols):
            dist = preds[len(seq), i, :].softmax(dim=0).detach().cpu().numpy()
            dist = np.concatenate((
                dist[:self.n_symbols],
                dist[self.eos:self.eos+1]
            ))
            dists.append(dist)
        return dists

    def predict_two(self, symbols, n_samples, data_rand):
        assert len(symbols) == 1
        next_dist, = self.predict_one(symbols)
        next_dist /= next_dist.sum()
        out = []
        samps = data_rand.choice(self.n_symbols+1, p=next_dist, size=n_samples)
        nsymbols = [symbols[0] + (samp,) for samp in samps]
        return self.predict_one(nsymbols)

    #def predict(self, symbols, eval_indices=None):
    #    seqs = [
    #        torch.tensor((self.sos,) + seq + (self.eos,)) for seq in symbols
    #    ]
    #    batch = pad_sequence(seqs, padding_value=self.pad).cuda()
    #    loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad, reduction="none")
    #    batch_in = batch[:-1, :]
    #    batch_out = batch[1:, :].view(-1)
    #    with torch.no_grad():
    #        preds, _ = self(batch_in)
    #        preds = preds.view(-1, self.n_vocab)
    #        loss = loss_fn(preds, batch_out).view(-1, batch.shape[1])

    #        ppl = 0
    #        count = 0
    #        for i in range(loss.shape[1]):
    #            if eval_indices is None:
    #                seq_indices = list(range(len(seqs[i])+1))
    #            else:
    #                seq_indices = eval_indices[i]
    #            for j in seq_indices:
    #                ppl += loss[j][i]
    #                count += 1
    #    return ppl / count

    @classmethod
    def train_model(cls, dfa, data_rand, n_embed, n_hidden, max_iters=2000,
            model_type="lstm",
            symbol_swap=0,
            symbol_mask=0,
            symbol_dropout=0,
            state_reset=0,
            state_skip=0,
            state_noise=0,
            state_dropout=0,
    ):
        rnn = SequenceModel(
            dfa.n_states, dfa.n_symbols, n_embed, n_hidden, model_type,
            symbol_dropout, state_dropout
        ).cuda()
        rnn.train()
        opt = optim.Adam(rnn.parameters(), lr=0.0003)
        loss_fn = nn.CrossEntropyLoss(ignore_index=rnn.pad)

        def inject_noise(seqs):
            noisy_seqs = []
            if symbol_swap == 0 and symbol_mask == 0:
                return seqs
            assert symbol_swap == 0 or symbol_mask == 0
            noise = max(symbol_swap, symbol_mask)
            for seq in seqs:
                for j in range(len(seq)):
                    if symbol_swap > 0:
                        new_symbol = data_rand.choice(dfa.n_symbols)
                    else:
                        new_symbol = rnn.pad
                    if data_rand.rand() < noise:
                        seq = seq[:j] + (new_symbol,) + seq[j+1:]
                noisy_seqs.append(seq)
            return noisy_seqs

        def make_batch(dfa, batch_size):
            seqs = [dfa.sample(data_rand) for _ in range(batch_size)]
            noisy_seqs = inject_noise(seqs)
            seqs = [torch.tensor((rnn.sos,) + seq + (rnn.eos,)) for seq in seqs]
            noisy_seqs = [torch.tensor((rnn.sos,) + seq + (rnn.eos,)) for seq in noisy_seqs]
            padded = pad_sequence(seqs, padding_value=rnn.pad)
            noisy_padded = pad_sequence(noisy_seqs, padding_value=rnn.pad)
            return noisy_padded.cuda(), padded.cuda()

        epoch_loss = 0
        for i in range(max_iters):
            batch_in, batch_out = make_batch(dfa, 64)
            batch_in = batch_in[:-1, :]
            batch_out = batch_out[1:, :].view(-1)
            preds, _ = rnn(
                batch_in,
                state_reset=state_reset,
                state_skip=state_skip,
                state_noise=state_noise
            )
            preds = preds.view(-1, rnn.n_vocab)
            loss = loss_fn(preds, batch_out)
            epoch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (i+1) % 100 == 0:
                epoch_loss /= 100
                best = dfa.approx_ppl
                tol = epoch_loss / best
                if tol < 1.005:
                    break
                epoch_loss = 0

        #print("finished training after", i, "epochs with tolerance", tol)
        rnn.eval()
        return rnn

class GaussianProbe(object):
    def __init__(self, means, covs):
        self.means = means
        self.covs = covs

    @classmethod
    def _run_rnn(cls, rnn, symbols):
        seqs = [torch.tensor((rnn.sos,) + seq + (rnn.eos,)) for seq in symbols]
        batch = pad_sequence(seqs, padding_value=rnn.pad).cuda()[:-1, :]
        _, hiddens = rnn(batch)
        hiddens = [hiddens[:len(s)+1, i, :] for i, s in enumerate(symbols)]
        return hiddens

    def predict(self, rnn, symbols):
        hiddens = self._run_rnn(rnn, symbols)
        predictions = []
        for i, seq in enumerate(symbols):
            seq_predictions = []
            for j in range(len(seq)+1):
                errs = [m - hiddens[i][j, :] for m in self.means]
                dists = [torch.norm(e).item() for e in errs]
                prediction = np.argmin(dists)
                seq_predictions.append(prediction)
            predictions.append(seq_predictions)
        return predictions

    def eval(self, rnn, symbols, states, eval_indices=None):
        hiddens = self._run_rnn(rnn, symbols)
        correct = 0
        error = 0
        confidence = 0
        total = 0
        for i, (seq, ann) in enumerate(zip(symbols, states)):
            if eval_indices is None:
                seq_eval_indices = list(range(len(ann)))
            else:
                seq_eval_indices = eval_indices[i]
            for j in seq_eval_indices:
                errs = [m - hiddens[i][j, :] for m in self.means]
                dists = [torch.norm(e).item() for e in errs]
                prediction = np.argmin(dists)
                scores = np.exp(-np.array(dists))
                error += dists[prediction]
                confidence += scores[prediction] / np.sum(scores)
                if prediction == ann[j]:
                    correct += 1
                total += 1

        stats = {
            "accuracy": correct / total,
            "confidence": confidence / total,
            "error": error / total,
        }

        return stats

    @classmethod
    def train(cls, dfa, rnn, symbols):
        hiddens = cls._run_rnn(rnn, symbols)

        buckets = [[] for _ in range(rnn.n_states)]
        for i, seq in enumerate(symbols):
            ann = dfa.annotate(seq)
            for j, state in enumerate(ann):
                buckets[state].append(hiddens[i][j, :])

        means = []
        covs = []
        for bucket in buckets:
            if len(bucket) == 0:
                mean = torch.ones(256).cuda() * 999
                cov = torch.ones(256, 256).cuda() * 999
            else:
                bucket_data = torch.stack(bucket)
                mean = torch.mean(bucket_data, dim=0)
                cov = (bucket_data.t() @ bucket_data) / bucket_data.shape[0]
            means.append(mean)
            covs.append(cov)

        return GaussianProbe(means, covs)

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CustomTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
            activation="relu", head_dropout=0):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.head_dropout = head_dropout
        #print(self.head_dropout)

        self.nhead = nhead

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.train and self.head_dropout > 0:
            src_mask = src_mask.unsqueeze(0).unsqueeze(1).expand(src.shape[1], self.nhead, -1, -1)
            src_mask = src_mask.clone()
            for i in range(self.nhead):
                if torch.rand(1).item() < self.head_dropout:
                    src_mask[:, i, :, :] = -np.inf
            for i in range(src_mask.shape[2]):
                src_mask[:, :, i, i] = 0
            src_mask = src_mask.view(src.shape[1] * self.nhead, src_mask.shape[2], src_mask.shape[3])
            #print(src.shape, self.nhead)
            #print(src_mask.shape)

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
