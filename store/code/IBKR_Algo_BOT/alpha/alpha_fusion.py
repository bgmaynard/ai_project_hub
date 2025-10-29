from __future__ import annotations
from dataclasses import dataclass
from math import exp, isfinite

EPS = 1e-12

@dataclass
class Ewma:
    alpha: float
    value: float | None = None
    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha*x + (1-self.alpha)*self.value
        return self.value if isfinite(self.value) else 0.0

class RunningWelford:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    @property
    def var(self) -> float:
        return self.M2 / max(1, self.n-1)

@dataclass
class AlphaFusionConfig:
    # hyperparams (simple defaults)
    alpha_vol: float = 0.2     # EWMA for |r|
    alpha_drift: float = 0.05  # EWMA for r
    learn_rate: float = 0.02
    l2: float = 0.001
    bins: int = 10             # for simple recalibration
    sim_c: float = 0.4

class AlphaFusion:
    """
    Minimal online model:
    - features: imb, mom (vol-normed r), drift, sentiment, barrier, vwapd, spr
    - logistic   p_model = sigmoid(beta·x)
    - calibration p_cal   = p_model * (hit_rate / mean_p)  (coarse bins)
    - reliability R       = map from rolling Brier to [0,1]
    - similarity M_t      = exp( c * (winrate - 0.5) )
    final p_final = clip( p_cal * R * M_t )
    """
    def __init__(self, cfg: AlphaFusionConfig | None = None) -> None:
        self.cfg = cfg or AlphaFusionConfig()
        # weights (start small)
        self.beta = [0.0]*8  # bias + 7 features
        self.ew_vol = Ewma(self.cfg.alpha_vol)
        self.ew_drift = Ewma(self.cfg.alpha_drift)
        self.brier_ew = Ewma(0.1)
        # simple bins for calibration
        self.cal_bins = [ {"sum_p":0.0,"sum_y":0.0,"n":0} for _ in range(self.cfg.bins) ]

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z > 20: return 1.0
        if z < -20: return 0.0
        return 1/(1+exp(-z))

    @staticmethod
    def _clip01(x: float) -> float:
        return 0.0 if x < 0 else 1.0 if x > 1 else x

    def _bin_ix(self, p: float) -> int:
        p = max(0.0, min(0.999999, p))
        return int(p * self.cfg.bins)

    def features_from_l1(self, bid: float, ask: float, bidSz: float, askSz: float,
                         last_mid: float | None, vwap: float | None, sentiment: float | None = 0.0):
        m = (bid+ask)/2 if bid and ask else (last_mid or 0.0)
        r = 0.0 if not last_mid else (m - last_mid)/(last_mid + EPS)
        vol = self.ew_vol.update(abs(r)) or EPS
        drift = self.ew_drift.update(r) or 0.0
        spr = (ask - bid)/max(m, EPS) if ask and bid and m>0 else 0.0
        vwapd = ( (m - (vwap or m)) / (0.001*max(m,EPS)) ) if vwap else 0.0
        D_B, D_A = float(bidSz or 0.0), float(askSz or 0.0)
        imb = (D_B - D_A)/max(D_B + D_A + EPS, EPS)
        mom = r/max(vol, EPS)
        # round/half barrier strength from fractional part
        frac = m - int(m) if m>0 else 0.0
        p00 = 1 - abs(frac - 0.00)/0.05
        p50 = 1 - abs(frac - 0.50)/0.05
        barrier = max(0.0, min(1.0, max(p00, p50)))
        s = float(sentiment or 0.0)
        return {
            "bias": 1.0,
            "imb": imb, "mom": mom, "drift": drift, "sent": s,
            "barrier": barrier, "vwapd": vwapd, "spr": spr
        }

    def predict_raw(self, feats: dict) -> float:
        x = [feats.get("bias",1.0), feats["imb"], feats["mom"], feats["drift"],
             feats["sent"], feats["barrier"], feats["vwapd"], feats["spr"]]
        z = sum(b*w for b,w in zip(self.beta, x))
        return self._sigmoid(z)

    def calibrate(self, p: float) -> float:
        b = self._bin_ix(p)
        bin_ = self.cal_bins[b]
        if bin_["n"] >= 25 and bin_["sum_p"]>0:
            mean_p = bin_["sum_p"]/bin_["n"]
            hit    = bin_["sum_y"]/max(1,bin_["n"])
            return self._clip01(p * (hit/max(mean_p, EPS)))
        return p

    def reliability(self, p: float, y: float | None = None) -> float:
        # update brier ewma when y provided (training calls)
        if y is not None:
            brier = (p - y)**2
            self.brier_ew.update(brier)
        e = self.brier_ew.value if self.brier_ew.value is not None else 0.05
        R = max(0.0, min(1.0, 1 - 2*e))
        return R

    def similarity_multiplier(self, winrate: float | None = None) -> float:
        if winrate is None: return 1.0
        return exp(self.cfg.sim_c * (winrate - 0.5))

    def fused_probability(self, feats: dict, winrate: float | None = None) -> float:
        p_model = self.predict_raw(feats)
        p_cal   = self.calibrate(p_model)
        R       = self.reliability(p_cal)
        M       = self.similarity_multiplier(winrate)
        return self._clip01(p_cal * R * M)

    # training hooks (optional usage)
    def observe(self, p: float, y: float) -> None:
        b = self._bin_ix(p)
        bin_ = self.cal_bins[b]
        bin_["sum_p"] += p
        bin_["sum_y"] += y
        bin_["n"]     += 1
        self.reliability(p, y)
