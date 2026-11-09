# -*- coding: utf-8 -*-
"""Numpy 64->64->32->1 ReLU MLP (6,273 params, same architecture/optimizer as
train_modules.py: AdamW lr=1e-3, batch 64, <=30 epochs) with BCE or pairwise-logistic loss.

Numpy instead of torch so the trainer is deterministic, dependency-light and unit-testable
with fake data. Train-only normalization statistics are stored *inside* the artifact
(the existing .pt artifacts did not store them)."""
from __future__ import annotations
import numpy as np

SIZES = [(64, 64), (64, 32), (32, 1)]


def n_params():
    return sum(i * o + o for i, o in SIZES)


class MLP:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i, o in SIZES:  # torch.nn.Linear default init range
            bound = 1.0 / np.sqrt(i)
            self.W.append(rng.uniform(-bound, bound, (i, o)))
            self.b.append(rng.uniform(-bound, bound, (o,)))
        self.mu = np.zeros(64)
        self.sd = np.ones(64)

    # -- forward/backward
    def _fwd(self, X):
        hs = [X]
        h = X
        for k in range(3):
            z = h @ self.W[k] + self.b[k]
            h = np.maximum(z, 0) if k < 2 else z
            hs.append(h)
        return hs

    def logits(self, X):
        Xn = (np.asarray(X, float) - self.mu) / self.sd
        return self._fwd(Xn)[-1][:, 0]

    def _grads(self, hs, dout):
        gW, gb = [None] * 3, [None] * 3
        d = dout[:, None]
        for k in (2, 1, 0):
            gW[k] = hs[k].T @ d
            gb[k] = d.sum(0)
            if k > 0:
                d = (d @ self.W[k].T) * (hs[k] > 0)
        return gW, gb

    def to_dict(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b],
                "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(0)
        m.W = [np.array(w) for w in d["W"]]
        m.b = [np.array(b) for b in d["b"]]
        m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        return m


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class _AdamW:
    def __init__(self, m, lr=1e-3, wd=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.lr, self.wd, self.b1, self.b2, self.eps, self.t = m, lr, wd, b1, b2, eps, 0
        self.s = [[np.zeros_like(p) for p in m.W + m.b] for _ in range(2)]

    def step(self, gW, gb):
        self.t += 1
        params = self.m.W + self.m.b
        for i, (p, g) in enumerate(zip(params, gW + gb)):
            p *= 1 - self.lr * self.wd
            self.s[0][i] = self.b1 * self.s[0][i] + (1 - self.b1) * g
            self.s[1][i] = self.b2 * self.s[1][i] + (1 - self.b2) * g * g
            mh = self.s[0][i] / (1 - self.b1 ** self.t)
            vh = self.s[1][i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _fit_norm(m, X):
    m.mu = X.mean(0)
    m.sd = X.std(0) + 1e-6


def train_bce(X, y, seed, epochs=30, bs=64, lr=1e-3):
    X, y = np.asarray(X, float), np.asarray(y, float)
    m = MLP(seed)
    _fit_norm(m, X)
    Xn = (X - m.mu) / m.sd
    opt, rng, losses = _AdamW(m, lr), np.random.default_rng(seed), []
    for _ in range(epochs):
        perm, tot = rng.permutation(len(X)), 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            hs = m._fwd(Xn[idx])
            p = _sig(hs[-1][:, 0])
            tot += float(-np.mean(y[idx] * np.log(p + 1e-12) + (1 - y[idx]) * np.log(1 - p + 1e-12))) * len(idx)
            opt.step(*m._grads(hs, (p - y[idx]) / len(idx)))
        losses.append(tot / len(X))
    return m, {"loss_curve": [round(l, 5) for l in losses], "n": int(len(X)), "steps": opt.t}


def train_pairwise(Xpos, Xneg, seed, epochs=30, bs=64, lr=1e-3):
    """L_Q = -log sigmoid(s(q+) - s(q-)) (HumanLoop §8.3)."""
    Xp, Xm = np.asarray(Xpos, float), np.asarray(Xneg, float)
    m = MLP(seed)
    _fit_norm(m, np.vstack([Xp, Xm]))
    Pn, Nn = (Xp - m.mu) / m.sd, (Xm - m.mu) / m.sd
    opt, rng, losses = _AdamW(m, lr), np.random.default_rng(seed), []
    for _ in range(epochs):
        perm, tot = rng.permutation(len(Xp)), 0.0
        for i in range(0, len(Xp), bs):
            idx = perm[i:i + bs]
            hp, hn = m._fwd(Pn[idx]), m._fwd(Nn[idx])
            diff = hp[-1][:, 0] - hn[-1][:, 0]
            tot += float(np.mean(np.logaddexp(0, -diff))) * len(idx)
            d = -(1 - _sig(diff)) / len(idx)
            gWp, gbp = m._grads(hp, d)
            gWn, gbn = m._grads(hn, -d)
            opt.step([a + b for a, b in zip(gWp, gWn)], [a + b for a, b in zip(gbp, gbn)])
        losses.append(tot / len(Xp))
    return m, {"loss_curve": [round(l, 5) for l in losses], "n_pairs": int(len(Xp)), "steps": opt.t}


def fit_platt(logits, y, iters=500, lr=0.05):
    """Dev-only probability calibration p = sigmoid(a*logit + b); MLP weights untouched."""
    z, y = np.asarray(logits, float), np.asarray(y, float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = _sig(a * z + b)
        a -= lr * float(np.mean((p - y) * z))
        b -= lr * float(np.mean(p - y))
    return {"a": a, "b": b, "n": int(len(z))}
