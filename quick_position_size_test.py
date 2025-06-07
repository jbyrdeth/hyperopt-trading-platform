#!/usr/bin/env python3
"""
Quick Position Size Test
Debug the astronomical position sizing issue
"""

import sys
import os
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime

# Test position size calculation
def test_position_sizing():
    print("🔍 POSITION SIZE CALCULATION TEST")
    print("=" * 50)
    
    # Simulate the position sizing calculation
    available_capital = 100000  # $100k
    position_size_pct = 0.95    # 95%
    current_price = 45000       # $45k per unit
    signal_strength = 1.0       # Maximum strength
    volatility = 0.025          # 2.5%
    
    print(f"Available capital: ${available_capital:,.2f}")
    print(f"Position size %: {position_size_pct:.1%}")
    print(f"Current price: ${current_price:,.2f}")
    print(f"Signal strength: {signal_strength:.2f}")
    print(f"Volatility: {volatility:.1%}")
    
    # Calculate position size like MovingAverageCrossover
    base_size = available_capital * position_size_pct / current_price
    adjusted_size = base_size * signal_strength
    volatility_adjustment = max(0.5, 1.0 - (volatility * 2))
    final_size = adjusted_size * volatility_adjustment
    
    print(f"\n📊 CALCULATION STEPS:")
    print(f"1. Base size: {available_capital} * {position_size_pct} / {current_price} = {base_size:.6f} units")
    print(f"2. Adjusted for signal: {base_size:.6f} * {signal_strength} = {adjusted_size:.6f} units")
    print(f"3. Volatility adjustment: max(0.5, 1.0 - ({volatility} * 2)) = {volatility_adjustment:.6f}")
    print(f"4. Final size: {adjusted_size:.6f} * {volatility_adjustment:.6f} = {final_size:.6f} units")
    
    # Calculate trade value
    trade_value = abs(final_size * current_price)
    position_pct_of_capital = trade_value / available_capital
    
    print(f"\n💰 TRADE DETAILS:")
    print(f"Final position size: {final_size:.6f} units")
    print(f"Trade value: {final_size:.6f} * ${current_price:,.2f} = ${trade_value:,.2f}")
    print(f"Position as % of capital: {position_pct_of_capital:.1%}")
    
    # Test with very low capital (simulating capital depletion)
    print(f"\n🚨 TESTING WITH DEPLETED CAPITAL:")
    low_capital = 0.01  # $0.01
    
    base_size_low = low_capital * position_size_pct / current_price
    adjusted_size_low = base_size_low * signal_strength
    final_size_low = adjusted_size_low * volatility_adjustment
    trade_value_low = abs(final_size_low * current_price)
    position_pct_low = trade_value_low / low_capital
    
    print(f"Available capital: ${low_capital:.2f}")
    print(f"Final size: {final_size_low:.10f} units")
    print(f"Trade value: ${trade_value_low:.10f}")
    print(f"Position as % of capital: {position_pct_low:.1%}")
    
    # Test what happens if capital goes negative
    print(f"\n🔥 TESTING WITH NEGATIVE CAPITAL:")
    negative_capital = -1000  # -$1000
    
    try:
        base_size_neg = negative_capital * position_size_pct / current_price
        adjusted_size_neg = base_size_neg * signal_strength
        final_size_neg = adjusted_size_neg * volatility_adjustment
        trade_value_neg = abs(final_size_neg * current_price)
        position_pct_neg = trade_value_neg / abs(negative_capital)
        
        print(f"Available capital: ${negative_capital:.2f}")
        print(f"Final size: {final_size_neg:.6f} units")
        print(f"Trade value: ${trade_value_neg:.2f}")
        print(f"Position as % of |capital|: {position_pct_neg:.1%}")
    except Exception as e:
        print(f"Error with negative capital: {e}")

if __name__ == "__main__":
    test_position_sizing() 