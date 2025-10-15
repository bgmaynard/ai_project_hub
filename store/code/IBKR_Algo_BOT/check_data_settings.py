"""
Data Settings Diagnostic Tool
=============================

This script checks what data is actually being downloaded
and helps identify the problem with the training pipeline.

Run this first to verify the issue before training.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_data_download(symbol, period, interval):
    """Check what data yfinance actually downloads"""
    
    print(f"\n{'='*70}")
    print(f"CHECKING DATA DOWNLOAD FOR {symbol}")
    print(f"{'='*70}")
    print(f"Requested: period={period}, interval={interval}\n")
    
    # Download
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if df.empty:
        print("❌ No data downloaded!")
        return None
    
    # Flatten columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Analysis
    start_date = df.index[0]
    end_date = df.index[-1]
    days_covered = (end_date - start_date).days
    
    print(f"📊 RESULTS:")
    print(f"   Bars downloaded: {len(df)}")
    print(f"   Start date: {start_date}")
    print(f"   End date: {end_date}")
    print(f"   Days covered: {days_covered}")
    print(f"   Years covered: {days_covered/365:.2f}")
    
    # Expected calculations
    if interval == '1h':
        # Hourly: ~6.5 trading hours/day * 252 trading days/year
        expected_per_year = 6.5 * 252
        expected_2y = expected_per_year * 2
        print(f"\n📈 EXPECTATIONS FOR HOURLY:")
        print(f"   Expected per year: ~{expected_per_year:.0f} bars")
        print(f"   Expected for 2y: ~{expected_2y:.0f} bars")
        print(f"   Actual: {len(df)} bars")
        
        if len(df) >= 3800:
            print(f"   ✅ LOOKS GOOD - Close to 2 years hourly")
        elif len(df) >= 3000 and len(df) < 3800:
            print(f"   ⚠️  WARNING - Looks like 60 days of 5-min resampled")
        else:
            print(f"   ⚠️  WARNING - Unexpected bar count")
            
    elif interval == '5m':
        # 5-min: ~78 bars/day * trading days
        expected_per_day = 78
        expected_60d = expected_per_day * 60
        print(f"\n📈 EXPECTATIONS FOR 5-MIN:")
        print(f"   Expected per day: ~{expected_per_day} bars")
        print(f"   Expected for 60d: ~{expected_60d} bars")
        print(f"   Actual: {len(df)} bars")
        
        if len(df) >= 4000:
            print(f"   ✅ Looks like 60 days of 5-min")
        else:
            print(f"   ⚠️  Unexpected bar count")
    
    # Sample data
    print(f"\n📋 SAMPLE DATA (first 5 rows):")
    print(df.head())
    
    print(f"\n📋 SAMPLE DATA (last 5 rows):")
    print(df.tail())
    
    print(f"\n{'='*70}\n")
    
    return df


def diagnose_issue():
    """Run diagnostic on both settings"""
    
    print("\n" + "🔍"*35)
    print("DATA DOWNLOAD DIAGNOSTIC TOOL")
    print("🔍"*35 + "\n")
    
    print("Testing AAPL with different settings...\n")
    
    # Test 1: What user WANTS (2y hourly)
    print("TEST 1: What we WANT (2 years hourly)")
    print("-" * 70)
    df_want = check_data_download('AAPL', period='2y', interval='1h')
    
    # Test 2: What user GETS (60d 5-min)
    print("TEST 2: What you might be GETTING (60 days 5-min)")
    print("-" * 70)
    df_get = check_data_download('AAPL', period='60d', interval='5m')
    
    # Compare
    print("="*70)
    print("COMPARISON")
    print("="*70)
    
    if df_want is not None and df_get is not None:
        print(f"\n2y hourly:   {len(df_want)} bars")
        print(f"60d 5-min:   {len(df_get)} bars")
        
        if len(df_want) < 3800:
            print("\n⚠️  PROBLEM DETECTED!")
            print("   Your '2y hourly' request is not returning 2 years of data!")
            print("   It's probably being limited by yfinance.")
            print("\n   SOLUTION:")
            print("   Use the train_real_stocks_MTF_FIXED.py script")
            print("   It explicitly sets: period='2y', interval='1h'")
        elif len(df_want) >= 3800:
            print("\n✅ GOOD NEWS!")
            print("   Your system CAN download 2 years hourly correctly")
            print("   The issue is in your training script settings")
            print("\n   SOLUTION:")
            print("   Replace train_real_stocks.py with train_real_stocks_MTF_FIXED.py")
    
    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\n🎯 NEXT STEPS:")
    print("\n1. Save the file train_real_stocks_MTF_FIXED.py")
    print("\n2. Run it with:")
    print("   python train_real_stocks_MTF_FIXED.py")
    print("\n3. Verify you see ~4032 bars in the output")
    print("\n4. You should get positive returns with MTF features!")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    diagnose_issue()
