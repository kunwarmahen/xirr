import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def xirr(cashflows):
    """
    Calculate the Extended Internal Rate of Return (XIRR) for a series of cashflows.
    
    Parameters:
    cashflows: DataFrame with 'date' and 'amount' columns
    
    Returns:
    xirr_value: The annualized internal rate of return
    """
    if len(cashflows) < 2:
        return None
        
    def xnpv(rate, cashflows):
        # Convert dates to days from first date
        t0 = pd.Timestamp(cashflows['date'].iloc[0])
        days = [(pd.Timestamp(d) - t0).days for d in cashflows['date']]
        return sum(cf / (1 + rate) ** (d / 365.0) for cf, d in zip(cashflows['amount'], days))
    
    # Find rate that gives xnpv of zero
    try:
        return optimize.newton(lambda r: xnpv(r, cashflows), 0.1)
    except RuntimeError:
        # If Newton's method fails, try with brentq
        try:
            return optimize.brentq(lambda r: xnpv(r, cashflows), -0.999, 10)
        except (ValueError, RuntimeError):
            # If brentq fails, try with other bounds or methods
            try:
                return optimize.brentq(lambda r: xnpv(r, cashflows), -0.9, 0.9)
            except (ValueError, RuntimeError):
                print("Warning: XIRR calculation failed to converge")
                return None

def calculate_investment_return(ticker, start_date, end_date, initial_investment=10000, annual_investment=1000):
    """
    Calculate the XIRR for an investment in a specific asset.
    
    Parameters:
    ticker: The ticker symbol for the asset
    start_date: The start date for the investment period
    end_date: The end date for the investment period
    initial_investment: The initial investment amount
    annual_investment: Amount to invest each year
    
    Returns:
    xirr_value: The annualized internal rate of return
    final_value: The final value of the investment
    cashflows: DataFrame with the cashflows used for the XIRR calculation
    total_invested: Total amount invested
    """
    try:
        # Download historical data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Check if data is empty properly
        if len(data) == 0:
            print(f"No data available for {ticker} in the specified date range.")
            return None, None, None, None
        
        # Create cashflows DataFrame starting with initial investment
        start_date_ts = pd.Timestamp(start_date)
        cashflows = pd.DataFrame({'date': [start_date], 'amount': [-initial_investment]})
        
        # Generate dates for annual investments
        annual_investment_dates = []
        
        # Start from one year after the initial investment
        current_date = start_date_ts + pd.DateOffset(years=1)
        end_date_ts = pd.Timestamp(end_date)
        
        # Add annual investments until we reach the end date
        while current_date < end_date_ts:
            annual_investment_dates.append(current_date)
            current_date = current_date + pd.DateOffset(years=1)
        
        # Create additional investments DataFrame
        additional_investments = pd.DataFrame({
            'date': annual_investment_dates,
            'amount': [annual_investment] * len(annual_investment_dates)
        }) if len(annual_investment_dates) > 0 else None
        
        # Calculate initial shares
        initial_price = data['Close'].iloc[0]
        shares = initial_investment / initial_price
        
        # Add additional investments to cashflows and calculate additional shares
        if additional_investments is not None:
            for _, row in additional_investments.iterrows():
                investment_date = row['date']
                
                # Add to cashflows
                cashflows = pd.concat([cashflows, pd.DataFrame({
                    'date': [investment_date.strftime('%Y-%m-%d')], 
                    'amount': [-row['amount']]
                })])
                
                # Find the closest trading day
                trading_dates = data.index[data.index >= investment_date]
                if len(trading_dates) > 0:
                    closest_date = trading_dates[0]
                    price_on_date = data.loc[closest_date, 'Close']
                    additional_shares = row['amount'] / price_on_date
                    shares += additional_shares
                else:
                    print(f"Warning: No trading data available after {investment_date} for {ticker}")
        
        # Calculate final value
        final_price = data['Close'].iloc[-1]
        final_value = shares * final_price
        
        # Add final cashflow (selling the investment)
        cashflows = pd.concat([cashflows, pd.DataFrame({'date': [end_date], 'amount': [final_value[ticker]]})])
        
        # Calculate XIRR
        xirr_value = xirr(cashflows)
        
        # Calculate total invested amount
        total_invested = initial_investment + (len(annual_investment_dates) * annual_investment if annual_investment_dates else 0)
        
        return xirr_value, final_value, cashflows, total_invested
    except Exception as e:
        print(f"Error in calculate_investment_return for {ticker}: {e}")
        return None, None, None, None

def calculate_multi_period_returns(ticker, end_date, periods, initial_investment=10000, annual_investment=1000):
    """
    Calculate returns for multiple time periods.
    
    Parameters:
    ticker: The ticker symbol for the asset
    end_date: The end date for all periods
    periods: List of periods in years to calculate returns for
    initial_investment: The initial investment amount
    annual_investment: Amount to invest each year
    
    Returns:
    Dictionary with period results
    """
    results = {}
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    for period in periods:
        try:
            # Calculate start date based on period
            start_date = end_date - timedelta(days=int(365.25*period))
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            print(f"Calculating {period}-year return for {ticker} ({start_date_str} to {end_date_str})...")
            
            xirr_value, final_value, cashflows, total_invested = calculate_investment_return(
                ticker, 
                start_date_str, 
                end_date_str, 
                initial_investment,
                annual_investment
            )
            
            if xirr_value is not None:
                results[period] = {
                    'XIRR': xirr_value * 100,  # Convert to percentage
                    'Final Value': final_value[ticker],
                    'Total Invested': total_invested,
                    'Total Return': (final_value[ticker] / total_invested - 1) * 100,  # Convert to percentage
                    'Start Date': start_date_str,
                    'End Date': end_date_str
                }
                print(f"  {period}-year XIRR: {xirr_value * 100:.2f}%, Final Value: ${final_value[ticker]:.2f}, Total Return: {(final_value[ticker] / total_invested - 1) * 100:.2f}%")
            else:
                print(f"  Could not calculate {period}-year return for {ticker}")
        except Exception as e:
            traceback.print_exc()
            print(f"  Error calculating {period}-year return for {ticker}: {e}")
    
    return results

def calculate_specific_date_range_returns(ticker, date_ranges, initial_investment=10000, annual_investment=1000):
    """
    Calculate returns for specific date ranges.
    
    Parameters:
    ticker: The ticker symbol for the asset
    date_ranges: Dictionary with names and (start_date, end_date) tuples
    initial_investment: The initial investment amount
    annual_investment: Amount to invest each year
    
    Returns:
    Dictionary with period results
    """
    results = {}
    
    for period_name, (start_date, end_date) in date_ranges.items():
        try:
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            print(f"Calculating return for {ticker} in {period_name} ({start_date_str} to {end_date_str})...")
            
            xirr_value, final_value, cashflows, total_invested = calculate_investment_return(
                ticker, 
                start_date_str, 
                end_date_str, 
                initial_investment,
                annual_investment
            )
            
            if xirr_value is not None:
                # Calculate duration in years for annual normalization
                duration_years = (end_date - start_date).days / 365.25
                
                results[period_name] = {
                    'XIRR': xirr_value * 100,  # Convert to percentage
                    'Final Value': final_value[ticker],
                    'Total Invested': total_invested,
                    'Total Return': (final_value[ticker] / total_invested - 1) * 100,  # Convert to percentage
                    'Start Date': start_date_str,
                    'End Date': end_date_str,
                    'Duration (Years)': duration_years
                }
                print(f"  {period_name} XIRR: {xirr_value * 100:.2f}%, Final Value: ${final_value[ticker]:.2f}, Total Return: {(final_value[ticker] / total_invested - 1) * 100:.2f}%")
            else:
                print(f"  Could not calculate return for {ticker} in {period_name}")
        except Exception as e:
            traceback.print_exc()
            print(f"  Error calculating return for {ticker} in {period_name}: {e}")
    
    return results

def create_decade_comparison_chart(decade_results):
    """
    Create charts comparing asset performance across different decades.
    
    Parameters:
    decade_results: Dictionary with results for each asset in each decade
    """
    # Extract data for plotting
    decades = list(list(decade_results.values())[0].keys())
    assets = list(decade_results.keys())
    
    # Colors for each asset
    colors = {
        'Gold': 'goldenrod',
        'S&P 500': 'darkblue',
        'Bitcoin': 'green'
    }
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    # Set width of bars
    bar_width = 0.25
    index = np.arange(len(decades))
    
    # Plot XIRR for each decade and asset
    for i, asset in enumerate(assets):
        xirr_values = []
        for decade in decades:
            if decade in decade_results[asset] and decade_results[asset][decade] is not None:
                xirr_values.append(decade_results[asset][decade]['XIRR'])
            else:
                xirr_values.append(np.nan)
        
        # Skip if all values are NaN (asset not available for any decade)
        if all(np.isnan(x) for x in xirr_values):
            continue
            
        bars = axs[0].bar(index + i*bar_width, xirr_values, bar_width, label=asset, color=colors.get(asset, f'C{i}'))
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                axs[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    axs[0].set_xlabel('Decade')
    axs[0].set_ylabel('XIRR (%)')
    axs[0].set_title('XIRR by Decade')
    axs[0].set_xticks(index + bar_width)
    axs[0].set_xticklabels(decades)
    axs[0].legend()
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot Total Return for each decade and asset
    for i, asset in enumerate(assets):
        total_returns = []
        for decade in decades:
            if decade in decade_results[asset] and decade_results[asset][decade] is not None:
                total_returns.append(decade_results[asset][decade]['Total Return'])
            else:
                total_returns.append(np.nan)
        
        # Skip if all values are NaN
        if all(np.isnan(x) for x in total_returns):
            continue
            
        bars = axs[1].bar(index + i*bar_width, total_returns, bar_width, label=asset, color=colors.get(asset, f'C{i}'))
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                axs[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    axs[1].set_xlabel('Decade')
    axs[1].set_ylabel('Total Return (%)')
    axs[1].set_title('Total Return by Decade')
    axs[1].set_xticks(index + bar_width)
    axs[1].set_xticklabels(decades)
    axs[1].legend()
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot Final Value for each decade and asset
    for i, asset in enumerate(assets):
        final_values = []
        for decade in decades:
            if decade in decade_results[asset] and decade_results[asset][decade] is not None:
                final_values.append(decade_results[asset][decade]['Final Value'])
            else:
                final_values.append(np.nan)
        
        # Skip if all values are NaN
        if all(np.isnan(x) for x in final_values):
            continue
            
        bars = axs[2].bar(index + i*bar_width, final_values, bar_width, label=asset, color=colors.get(asset, f'C{i}'))
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                axs[2].text(bar.get_x() + bar.get_width()/2., height + 500,
                        f'${height:.0f}', ha='center', va='bottom', fontsize=9)
    
    axs[2].set_xlabel('Decade')
    axs[2].set_ylabel('Final Value ($)')
    axs[2].set_title('Final Investment Value by Decade')
    axs[2].set_xticks(index + bar_width)
    axs[2].set_xticklabels(decades)
    axs[2].legend()
    axs[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create heatmap for XIRR comparison
    heatmap_data = np.zeros((len(assets), len(decades)))
    
    for i, asset in enumerate(assets):
        for j, decade in enumerate(decades):
            if decade in decade_results[asset] and decade_results[asset][decade] is not None:
                heatmap_data[i, j] = decade_results[asset][decade]['XIRR']
            else:
                heatmap_data[i, j] = np.nan
    
    # Create a mask for NaN values
    mask = np.isnan(heatmap_data)
    
    im = axs[3].imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=axs[3], label='XIRR (%)')
    
    # Add text annotations
    for i in range(len(assets)):
        for j in range(len(decades)):
            value = heatmap_data[i, j]
            if not np.isnan(value):
                axs[3].text(j, i, f'{value:.1f}%', ha='center', va='center', 
                         color='black' if 0 <= value < 20 else 'white')
    
    axs[3].set_xticks(np.arange(len(decades)))
    axs[3].set_yticks(np.arange(len(assets)))
    axs[3].set_xticklabels(decades)
    axs[3].set_yticklabels(assets)
    axs[3].set_title('XIRR Heatmap by Asset and Decade')
    
    plt.tight_layout()
    plt.savefig('xirrd_comparison.png')
    plt.show()

def main():
    
    end_date = datetime(2025, 4, 16)
    
    # Define assets to analyze
    assets = {
        'Gold': 'GLD',       # SPDR Gold Shares ETF
        'S&P 500': 'SPY',    # SPDR S&P 500 ETF
        'Bitcoin': 'BTC-USD', # Bitcoin USD
    }
    
    # Define periods to analyze (in years)
    periods = [1, 5, 10, 15, 20, 25, 50]
    
    # Define specific date ranges for decade analysis
    date_ranges = {
        '1991-2000': (datetime(1991, 1, 1), datetime(2000, 12, 31)),
        '2001-2010': (datetime(2001, 1, 1), datetime(2010, 12, 31)),
        '2011-2020': (datetime(2011, 1, 1), datetime(2020, 12, 31)),
        '2021-2025': (datetime(2021, 1, 1), datetime(2025, 4, 16))  
    }
    
    # Store all results
    year_results = {}
    decade_results = {}
    
    # PART 1: Calculate year-based returns
    print("\n" + "="*80)
    print("CALCULATING RETURNS FOR DIFFERENT YEAR PERIODS")
    print("="*80)
    
    for asset_name, ticker in assets.items():
        print(f"\nAnalyzing {asset_name} ({ticker}) across multiple time periods...")
        
        try:
            # Skip certain periods based on data availability
            asset_periods = periods.copy()
            if asset_name == 'Bitcoin':
                # Bitcoin data typically only available since ~2013
                asset_periods = [p for p in periods if p <= 15]
                
            if asset_name == 'Gold' and ticker == 'GLD':
                # GLD ETF started in 2004
                asset_periods = [p for p in periods if p <= 20]
                
            year_results[asset_name] = calculate_multi_period_returns(
                ticker, 
                end_date, 
                asset_periods,
                initial_investment=10000,
                annual_investment=1000
            )
            
        except Exception as e:
            print(f"Error analyzing {asset_name}: {e}")
    
    # PART 2: Calculate decade-based returns
    print("\n" + "="*80)
    print("CALCULATING RETURNS FOR SPECIFIC DECADES")
    print("="*80)
    
    for asset_name, ticker in assets.items():
        print(f"\nAnalyzing {asset_name} ({ticker}) for specific date ranges...")
        
        # Use appropriate ticker or proxy for each time period
        decade_results[asset_name] = {}
        
        # Define proxies for assets that don't have data for all periods
        proxies = {}
        
        if asset_name == 'Gold':
            # GLD ETF started in 2004, use ^GOLD or other proxy for earlier periods
            proxies = {
                '1991-2000': 'XAU',    # Philadelphia Gold and Silver Index as proxy
                '2001-2010': None,     # Will use GLD from 2004 onwards
                '2011-2020': None,     # Use default ticker
                '2021-2025': None      # Use default ticker
            }
        
        if asset_name == 'Bitcoin':
            # Bitcoin data only available since ~2013
            proxies = {
                '1991-2000': None,     # No data available
                '2001-2010': None,     # No data available
                '2011-2020': None,     # Use default ticker
                '2021-2025': None      # Use default ticker
            }
        
        for period_name, (start_date, end_date) in date_ranges.items():
            # Skip Bitcoin for periods before it existed
            if asset_name == 'Bitcoin' and start_date.year < 2010:
                print(f"  Skipping {period_name} for Bitcoin (not available)")
                decade_results[asset_name][period_name] = None
                continue
                
            # Skip GLD for periods before it existed, try to use proxy if available
            if asset_name == 'Gold' and start_date.year < 2004 and period_name == '1990-2000':
                print(f"  Gold ETF (GLD) not available for {period_name}, attempting to use proxy")
                current_ticker = proxies.get(period_name)
                print(current_ticker)
                if current_ticker is None:
                    print(f"  No proxy available for {asset_name} in {period_name}")
                    decade_results[asset_name][period_name] = None
                    continue
            else:
                current_ticker = ticker
            
            try:
                results = calculate_investment_return(
                    current_ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    initial_investment=10000,
                    annual_investment=1000
                )
                
                if results[0] is not None:  # Unpack the tuple
                    xirr_value, final_value, cashflows, total_invested = results
                    
                    decade_results[asset_name][period_name] = {
                        'XIRR': xirr_value * 100,  # Convert to percentage
                        'Final Value': final_value[ticker],
                        'Total Invested': total_invested,
                        'Total Return': (final_value[ticker] / total_invested - 1) * 100,  # Convert to percentage
                        'Start Date': start_date.strftime('%Y-%m-%d'),
                        'End Date': end_date.strftime('%Y-%m-%d'),
                        'Duration (Years)': (end_date - start_date).days / 365.25
                    }
                    
                    print(f"  {period_name} XIRR: {xirr_value * 100:.2f}%, Final Value: ${final_value[ticker]:.2f}, Total Return: {(final_value[ticker] / total_invested - 1) * 100:.2f}%")
                else:
                    print(f"  Could not calculate return for {asset_name} in {period_name}")
                    decade_results[asset_name][period_name] = None
                    
            except Exception as e:
                traceback.print_exc()
                print(f"  Error calculating return for {asset_name} in {period_name}: {e}")
                decade_results[asset_name][period_name] = None
    
    # Create comparison charts for decade analysis
    if decade_results:
        create_decade_comparison_chart(decade_results)
    
    # Create charts for year-based analysis
    if year_results:
        # XIRR Comparison across time periods
        plt.figure(figsize=(15, 8))
        
        # Prepare data for plotting
        bar_width = 0.2
        
        # Set position of bars on X axis
        common_periods = [p for p in [1, 5, 10] if any(p in year_results[asset] for asset in year_results)]
        positions = np.arange(len(common_periods))
        
        # Colors for each asset
        colors = {
            'Gold': 'goldenrod',
            'S&P 500': 'darkblue',
            'Bitcoin': 'green'
        }
        
        # Plot XIRR for each asset and common period
        i = 0
        for asset_name in year_results:
            # Extract XIRR values for common periods
            xirr_values = []
            for period in common_periods:
                if period in year_results[asset_name]:
                    xirr_values.append(year_results[asset_name][period]['XIRR'])
                else:
                    xirr_values.append(0)  # No data
            
            # Plot bars
            bars = plt.bar(positions + i*bar_width, xirr_values, bar_width, label=asset_name, color=colors[asset_name])
            
            # Add values on top of bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:  # Only add text if there's data
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
            
            i += 1
        
        # Add labels and legend
        plt.xlabel('Investment Period (Years)')
        plt.ylabel('XIRR (%)')
        plt.title('XIRR Comparison Across Time Periods')
        plt.xticks(positions + bar_width, common_periods)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('xirrd_comparison_periods.png')
        
        # Line chart for XIRR trends over time
        plt.figure(figsize=(12, 6))
        
        for asset_name in year_results:
            # Skip if no data
            if not year_results[asset_name]:
                continue
                
            periods = sorted(year_results[asset_name].keys())
            xirr_values = [year_results[asset_name][period]['XIRR'] for period in periods]
            
            plt.plot(periods, xirr_values, 'o-', label=asset_name, color=colors[asset_name])
            
            # Add values at each point
            for i, (period, xirr) in enumerate(zip(periods, xirr_values)):
                plt.text(period, xirr + 1, f'{xirr:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Investment Period (Years)')
        plt.ylabel('XIRR (%)')
        plt.title('XIRR Trends Over Different Time Periods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(sorted(list(set().union(*[year_results[asset].keys() for asset in year_results if year_results[asset]]))))
        
        plt.tight_layout()
        plt.savefig('xirrd_trends.png')
        
        # Print detailed results table
        print("\n" + "="*80)
        print("DETAILED RESULTS SUMMARY - YEAR-BASED ANALYSIS")
        print("="*80)
        
        for asset_name in year_results:
            print(f"\n{asset_name} Results:")
            print("-" * 80)
            print(f"{'Period (Years)':<15} {'XIRR':<15} {'Total Return':<15} {'Final Value':<15} {'Total Invested':<15}")
            print("-" * 80)
            
            periods = sorted(year_results[asset_name].keys())
            for period in periods:
                result = year_results[asset_name][period]
                print(f"{period:<15} {result['XIRR']:.2f}% {result['Total Return']:.2f}% ${result['Final Value']:.2f} ${result['Total Invested']:.2f}")
        
        print("\n" + "="*80)
        print("DETAILED RESULTS SUMMARY - DECADE-BASED ANALYSIS")
        print("="*80)
        
        for asset_name in decade_results:
            print(f"\n{asset_name} Results:")
            print("-" * 100)
            print(f"{'Period':<15} {'XIRR':<15} {'Total Return':<15} {'Final Value':<15} {'Total Invested':<15} {'Duration (Years)':<15}")
            print("-" * 100)
            
            for period_name in date_ranges.keys():
                if period_name in decade_results[asset_name] and decade_results[asset_name][period_name] is not None:
                    result = decade_results[asset_name][period_name]
                    print(f"{period_name:<15} {result['XIRR']:.2f}% {result['Total Return']:.2f}% ${result['Final Value']:.2f} ${result['Total Invested']:.2f} {result['Duration (Years)']:.1f}")
                else:
                    print(f"{period_name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        plt.show()

if __name__ == "__main__":
    main()