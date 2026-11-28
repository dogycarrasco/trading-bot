import os, sys
sys.path.insert(0, os.path.abspath("."))
from trading_bot.backtest.grid_search_optimize import count_walk_forward_windows

def test_walk_forward_windows():
    assert count_walk_forward_windows(61) >= 3

if __name__ == "__main__":
    test_walk_forward_windows()
    print("OK")
