#!/usr/bin/env python3
"""
Recovery script para Fix 4 + 5 + 6.

Uso:
    cd ~/trading-bot
    python recover_fix456.py
"""
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path

GITIGNORE = """\
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.venv/
venv/
env/
results/*.csv
results/*.json
results/*.png
!results/live_state_v8.json
!results/live_log_v8.txt
dead_code_audit_post.txt
"""

RISK_MODEL_MD = """\
# Modelo de riesgo del bot v35 ADAPTIVE

Este documento existe porque dos pares de constantes parecen inconsistentes al leer el codigo por primera vez. No lo son: cada par pertenece a una capa distinta con un proposito distinto.

## 1. SL/TP: triple barrier (backtest) vs trailing stop (live)

### Capa A - Backtest (risk/manager.py + features/triple_barrier.py)

```yaml
risk:
  stop_loss_atr_mult: 3.0
  take_profit_atr_mult: 10.0
  trailing_stop_enabled: true
  trailing_stop_trigger_atr: 3.0
  trailing_stop_distance_atr: 1.5