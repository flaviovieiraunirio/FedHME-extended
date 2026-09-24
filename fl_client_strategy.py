"""
fl_client_strategy.py
---------------------
COMPATIBILITY — this file only re-exports names so that legacy code
importing directly from fl_client_strategy keeps working.

After the refactor:
  - FedAvgClient (formerly TransitionFLClient) -> fl_algorithms.py
  - FedAvgStrategy (formerly Dimension2Strategy) -> fl_algorithms.py
  - Helpers (_seed_worker, _pos_to_cfg, etc.) -> fl_utils.py
  - make_client_fn -> fl_utils.py

This file can be removed once run_experiment.py no longer imports
from fl_client_strategy.
"""

# Re-exported for compatibility with existing imports
from fl_algorithms import FedAvgClient as TransitionFLClient
from fl_algorithms import FedAvgStrategy as Dimension2Strategy
from fl_algorithms import make_fedavg_client_fn as make_client_fn
from fl_utils import _seed_worker, _pos_to_cfg, _cfg_to_pos
