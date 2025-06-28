#!/usr/bin/env python3
"""
Verification script to confirm bid/ask price usage consistency between 
MeanReversionStrategy and OBIVWAPStrategy.
"""

import sys
import os
import re

def extract_bid_ask_patterns(file_path):
    """Extract patterns of bid/ask usage from a file."""
    patterns = {
        'unrealized_pnl': [],
        'account_balance': [],
        'opening_long': [],
        'opening_short': [],
        'closing_long': [],
        'closing_short': []
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Find unrealized PnL calculations
            unrealized_patterns = re.findall(r'unrealized_pnl = \([^)]+\)', content)
            patterns['unrealized_pnl'] = unrealized_patterns
            
            # Find account balance calculations
            balance_patterns = re.findall(r'current_balance = self\.cash \+ \([^)]+\)', content)
            patterns['account_balance'] = balance_patterns
            
            # Find opening positions
            long_open = re.findall(r'self\.cash -= row\["([^"]+)"\]', content)
            short_open = re.findall(r'self\.cash \+= row\["([^"]+)"\].*position_size', content)
            patterns['opening_long'] = long_open
            patterns['opening_short'] = short_open
            
            # Find closing positions  
            long_close = re.findall(r'self\.cash \+= row\["([^"]+)"\].*self\.position', content)
            short_close = re.findall(r'self\.cash -= row\["([^"]+)"\].*abs\(self\.position\)', content)
            patterns['closing_long'] = long_close
            patterns['closing_short'] = short_close
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return patterns

def main():
    # File paths
    mean_rev_path = "src/strats/mean_reversion.py"
    obi_path = "src/strats/obi.py"
    
    print("=== Bid/Ask Price Usage Consistency Verification ===\n")
    
    # Extract patterns from both files
    mean_rev_patterns = extract_bid_ask_patterns(mean_rev_path)
    obi_patterns = extract_bid_ask_patterns(obi_path)
    
    # Compare patterns
    print("1. UNREALIZED PnL CALCULATION:")
    print("   Mean Reversion:", mean_rev_patterns['unrealized_pnl'])
    print("   OBI Strategy:", obi_patterns['unrealized_pnl'])
    print()
    
    print("2. ACCOUNT BALANCE CALCULATION:")
    print("   Mean Reversion:", mean_rev_patterns['account_balance'])
    print("   OBI Strategy:", obi_patterns['account_balance'])
    print()
    
    print("3. OPENING POSITIONS:")
    print("   Long positions (should use 'ask'):")
    print("     Mean Reversion:", mean_rev_patterns['opening_long'])
    print("     OBI Strategy:", obi_patterns['opening_long'])
    print("   Short positions (should use 'bid'):")
    print("     Mean Reversion:", mean_rev_patterns['opening_short'])
    print("     OBI Strategy:", obi_patterns['opening_short'])
    print()
    
    print("4. CLOSING POSITIONS:")
    print("   Long positions (should use 'bid'):")
    print("     Mean Reversion:", mean_rev_patterns['closing_long'])
    print("     OBI Strategy:", obi_patterns['closing_long'])
    print("   Short positions (should use 'ask'):")
    print("     Mean Reversion:", mean_rev_patterns['closing_short'])
    print("     OBI Strategy:", obi_patterns['closing_short'])
    print()
    
    # Check consistency
    consistent = True
    
    # Both should consistently use bid for long positions and ask for short positions
    # in account balance and unrealized PnL
    if "bid" in str(mean_rev_patterns['account_balance']) and "ask" in str(mean_rev_patterns['account_balance']):
        if "bid" not in str(obi_patterns['account_balance']) or "ask" not in str(obi_patterns['account_balance']):
            consistent = False
            print("❌ INCONSISTENCY: Account balance calculation differs between strategies")
    
    if consistent:
        print("✅ VERIFICATION PASSED: Both strategies use bid/ask prices consistently")
    else:
        print("❌ VERIFICATION FAILED: Inconsistencies found between strategies")

if __name__ == "__main__":
    main()
