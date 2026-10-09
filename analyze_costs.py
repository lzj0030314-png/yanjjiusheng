# -*- coding: utf-8 -*-
"""分析目标成本，反推正确的成本参数"""
import numpy as np
from scipy.stats import norm

# 目标值
targets = {
    'Struct':  {'floor': [0.2898982393, 0.2466687704, 0.1241894602], 'total': 0.66076},
    'Disp':    {'floor': [0.69910268, 0.6597298695, 0.2754374505], 'total': 1.63427},
    'Accel':   {'floor': [0.000000, 0.014341, 0.08605108], 'total': 0.10039},
}

bldg_cost = 111.50 * 3000  # 334500

# EDP 84th percentile (approximate from 11 waves)
# Floor 1, 2, 3 X-direction drifts
drifts = {
    1: [0.00261499, 0.00602316, 0.00207613, 0.00712557, 0.00534886, 0.00638162, 0.00658842, 0.00599932, 0.00269024, 0.00238178, 0.00356173],
    2: [0.00345293, 0.0062728, 0.00276707, 0.00702827, 0.00557371, 0.00618857, 0.0066805, 0.00632475, 0.00358628, 0.00314373, 0.00454008],
    3: [0.00241831, 0.00419403, 0.00209482, 0.0048087, 0.00373573, 0.00428653, 0.00479585, 0.00419548, 0.00261357, 0.00229793, 0.00341951],
}
accels_ms2 = {
    1: [2]*11,  # constant
    2: [3.45851, 6.26403, 3.08905, 7.31474, 5.69468, 5.06894, 6.61798, 6.01965, 4.3311, 3.88886, 6.17024],
    3: [4.19913, 7.58973, 4.15218, 7.86664, 6.44658, 6.74229, 8.28449, 7.15564, 5.14074, 5.09409, 6.17422],
}

def calc_84th_edp(values):
    """84th percentile of EDP = exp(mean(ln) + std(ln))"""
    vals = np.array(values, dtype=float)
    vals = vals[vals > 1e-10]
    if len(vals) == 0: return 0
    return float(np.exp(np.mean(np.log(vals)) + np.std(np.log(vals), ddof=1)))

print("=== 84th percentile EDPs ===")
for f in [1,2,3]:
    d84 = calc_84th_edp(drifts[f])
    a84 = calc_84th_edp(accels_ms2[f]) / 9.81  # to g
    print(f"Floor {f}: drift_84th={d84:.6f}, accel_84th={a84:.4f}g")

# Fragility params
fragility = {
    'Struct': {
        'medians': [0.0035, 0.0050, 0.0080, 0.0130],
        'betas': [0.37, 0.30, 0.32, 0.46],
        'loss_ratios': [0.0, 0.10, 0.20, 0.50, 1.00],
        'repair_factors': [0.0, 1.20, 1.07, 1.15, 3.57],
        'vol_discount': 0.85,
        'qty': 4,
        'edp': 'drift',
    },
    'Disp': {
        'medians': [0.0020, 0.0040, 0.0080],
        'betas': [0.30, 0.30, 0.30],
        'loss_ratios': [0.0, 0.50, 0.22, 1.00],
        'repair_factors': [0.0, 1.15, 1.21, 1.27],
        'vol_discount': 0.89,
        'qty': 37,
        'edp': 'drift',
    },
    'Accel': {
        'medians': [0.80, 1.10, 1.69],
        'betas': [0.38, 0.32, 0.25],
        'loss_ratios': [0.0, 0.50, 0.10, 1.00],
        'repair_factors': [0.0, 1.93, 1.49, 1.31],
        'vol_discount': 0.96,
        'qty': 81,
        'edp': 'accel',
    },
}

def calc_ds_probs(edp_val, medians, betas):
    n = len(medians)
    edp_val = max(edp_val, 1e-10)
    exceed = np.array([norm.cdf(np.log(edp_val / m) / b) for m, b in zip(medians, betas)])
    probs = np.zeros(n + 1)
    probs[0] = 1.0 - exceed[0]
    for j in range(n - 1):
        probs[j + 1] = exceed[j] - exceed[j + 1]
    probs[-1] = exceed[-1]
    probs = np.clip(probs, 0, 1)
    if probs.sum() > 0: probs /= probs.sum()
    return probs

print("\n=== Damage state probabilities at 84th percentile EDP ===")
for ct, db in fragility.items():
    for f in [1,2,3]:
        edp = calc_84th_edp(drifts[f]) if db['edp']=='drift' else calc_84th_edp(accels_ms2[f])/9.81
        probs = calc_ds_probs(edp, db['medians'], db['betas'])
        # Expected loss factor = sum(P(DS=j) * loss[j] * repair[j] * vol)
        exp_factor = sum(probs[j] * db['loss_ratios'][j] * db['repair_factors'][j] * db['vol_discount'] for j in range(len(probs)))
        target_pct = targets[ct]['floor'][f-1]
        target_cost = target_pct / 100 * bldg_cost
        # cost * qty * exp_factor = target_cost
        if db['qty'] * exp_factor > 0:
            implied_cost = target_cost / (db['qty'] * exp_factor)
        else:
            implied_cost = float('inf')
        print(f"  {ct} Floor {f}: EDP={edp:.6f}, probs={[f'{p:.3f}' for p in probs]}, "
              f"exp_factor={exp_factor:.4f}, target={target_pct:.6f}%, "
              f"implied_cost_per_unit={implied_cost:.1f}")
    print()

# Also check with stochastic approach: simulate 1000 samples and take 84th percentile
print("\n=== Stochastic simulation approach (1000 sims, 84th percentile) ===")
np.random.seed(42)
for ct, db in fragility.items():
    for f in [1,2,3]:
        # Generate 1000 EDP samples from lognormal
        if db['edp'] == 'drift':
            vals = np.array(drifts[f], dtype=float)
        else:
            vals = np.array(accels_ms2[f], dtype=float) / 9.81
        log_vals = np.log(vals[vals > 1e-10])
        mu, sigma = np.mean(log_vals), np.std(log_vals, ddof=1)
        sim_edps = np.exp(np.random.normal(mu, sigma, 1000))
        
        # For each sim, sample DS and compute cost
        costs = []
        for edp in sim_edps:
            probs = calc_ds_probs(edp, db['medians'], db['betas'])
            ds = np.random.choice(len(probs), p=probs)
            if ds > 0:
                cost = db['qty'] * 1.0 * db['loss_ratios'][ds] * db['repair_factors'][ds] * db['vol_discount']
            else:
                cost = 0
            costs.append(cost)
        
        # 84th percentile (lognormal fit on positive values)
        costs = np.array(costs)
        valid = costs[costs > 1e-10]
        if len(valid) > 0:
            pct84 = np.exp(np.mean(np.log(valid)) + np.std(np.log(valid), ddof=1))
        else:
            pct84 = 0
        
        target_pct = targets[ct]['floor'][f-1]
        target_cost = target_pct / 100 * bldg_cost
        implied_cost = target_cost / pct84 if pct84 > 0 else float('inf')
        print(f"  {ct} Floor {f}: 84th_cost_factor={pct84:.4f}, target={target_pct:.6f}%, "
              f"implied_cost_per_unit={implied_cost:.1f}")
    print()
