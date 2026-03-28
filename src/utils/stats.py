# src/utils/stats.py
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

def diff_ci(success_a, total_a, success_b, total_b):
    p1 = success_a/total_a; p2 = success_b/total_b
    diff = p1 - p2
    low, high = confint_proportions_2indep(success_a, total_a, success_b, total_b, method='wald')
    stat, p = proportions_ztest([success_a, success_b], [total_a, total_b])
    return {"diff": diff, "ci": (low, high), "p": p}