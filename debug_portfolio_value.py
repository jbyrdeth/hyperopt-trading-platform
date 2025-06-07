#!/usr/bin/env python3
"""
Debug Portfolio Value Calculation

Test to verify the portfolio value double-counting theory.
"""

def simulate_trade_sequence():
    """Simulate a simple trade sequence to show the double-counting issue."""
    
    print("🔍 PORTFOLIO VALUE DOUBLE-COUNTING SIMULATION")
    print("=" * 60)
    
    # Initial state
    initial_capital = 100000
    current_capital = initial_capital
    position_size = 0.0
    
    print(f"Initial state:")
    print(f"  Cash: ${current_capital:,.2f}")
    print(f"  Position: {position_size} units")
    print(f"  Total portfolio: ${current_capital:,.2f}")
    
    # Trade 1: Buy 2 units at $50,000
    trade_price = 50000
    trade_size = 2.0
    transaction_costs = 100  # Small cost
    
    print(f"\n📊 TRADE 1: Buy {trade_size} units at ${trade_price:,.2f}")
    
    # OLD METHOD (incorrect - caused -100% returns):
    print(f"\n❌ OLD METHOD:")
    old_capital = current_capital - (trade_size * trade_price) - transaction_costs
    print(f"  Cash after trade: ${old_capital:,.2f}")
    print(f"  Position: {trade_size} units")
    old_portfolio = old_capital + (trade_size * trade_price)
    print(f"  Portfolio value: ${old_portfolio:,.2f}")
    
    # NEW METHOD (our current fix):
    print(f"\n✅ NEW METHOD:")
    new_capital = current_capital - transaction_costs  # Only deduct costs
    print(f"  Cash after trade: ${new_capital:,.2f}")
    print(f"  Position: {trade_size} units")
    new_portfolio = new_capital + (trade_size * trade_price)
    print(f"  Portfolio value: ${new_portfolio:,.2f}")
    
    print(f"\n⚠️ ISSUE: New method double-counts the asset value!")
    print(f"  Expected portfolio value: ${initial_capital - transaction_costs:,.2f}")
    print(f"  Actual portfolio value: ${new_portfolio:,.2f}")
    print(f"  Difference: ${new_portfolio - (initial_capital - transaction_costs):,.2f}")
    
    # Price movement simulation
    print(f"\n📈 PRICE MOVEMENT: ${trade_price:,.2f} → ${trade_price+1000:,.2f}")
    new_price = trade_price + 1000
    
    print(f"\n❌ Current calculation (WRONG):")
    wrong_portfolio = new_capital + (trade_size * new_price)
    print(f"  Portfolio value: ${wrong_portfolio:,.2f}")
    print(f"  Return: {(wrong_portfolio/initial_capital - 1)*100:.2f}%")
    
    print(f"\n✅ Correct calculation:")
    # We should track the asset value separately
    asset_value = trade_size * new_price
    available_cash = new_capital  # This is NOT the total cash equivalent
    
    # Correct approach: Portfolio = Initial capital + P&L
    unrealized_pnl = trade_size * (new_price - trade_price)
    correct_portfolio = initial_capital - transaction_costs + unrealized_pnl
    print(f"  Portfolio value: ${correct_portfolio:,.2f}")
    print(f"  Return: {(correct_portfolio/initial_capital - 1)*100:.2f}%")
    
    print(f"\n🎯 THE FIX:")
    print(f"  We need to track portfolio value as:")
    print(f"  Portfolio = Initial Capital + Total P&L - Total Costs")
    print(f"  NOT: Portfolio = Current Cash + Position Value")

if __name__ == "__main__":
    simulate_trade_sequence() 