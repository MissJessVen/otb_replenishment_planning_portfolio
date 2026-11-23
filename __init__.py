"""otb_planner - SKU-level forecasting, replenishment and open-to-buy for a
high-SKU, short-lifecycle assortment."""
from .config import V16, V17, PlannerConfig
from .pipeline import CycleResult, run_cycle

__all__ = ["PlannerConfig", "V16", "V17", "run_cycle", "CycleResult"]
__version__ = "1.6.0"
