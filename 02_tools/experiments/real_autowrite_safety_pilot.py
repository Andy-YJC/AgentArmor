"""Known bypasses plus previously rejected and passing normal controls."""
from experiments.real_route_support import execute
from experiments.real_autowrite import load as all_samples, evaluate

IDS={"AWE-001","AWE-020","AWF-009","AWF-013",
     "B-005","B-012","B-014","B-018","B-022","B-023","B-038",
     "B-001","B-002","B-003","B-004","B-006","B-007","B-008","B-009"}

def load():
    return [s for s in all_samples() if s["id"] in IDS]

if __name__=="__main__":execute("autowrite",load,evaluate)
