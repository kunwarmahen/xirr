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
        })
        
        # Calculate initial shares
        initial_price = data['Close'].iloc[0]
        shares = initial_investment / initial_price
        
        # Add additional investments to cashflows and calculate additional shares
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
        total_invested = initial_investment + len(annual_investment_dates) * annual_investment
        
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
                    'Total Return': (final_value[ticker] / total_invested - 1) * 100  # Convert to percentage
                }
                print(f"  {period}-year XIRR: {xirr_value * 100:.2f}%, Final Value: ${final_value[ticker]:.2f}, Total Return: {(final_value[ticker] / total_invested - 1) * 100:.2f}%")
            else:
                print(f"  Could not calculate {period}-year return for {ticker}")
        except Exception as e:
            traceback.print_exc()
            print(f"  Error calculating {period}-year return for {ticker}: {e}")
    
    return results

def main():
    
    end_date = datetime(2025, 4, 16)
    
    # Define assets to analyze
    assets = {
        'Gold': 'GLD',       # SPDR Gold Shares ETF
        'S&P 500': 'SPY',    # SPDR S&P 500 ETF
        'Bitcoin': 'BTC-USD' # Bitcoin USD
    }
    
    # Define periods to analyze (in years)
    periods = [1, 5, 10, 15, 20, 25, 50]
    
    # Store all results
    all_results = {}
    
    # Calculate returns for each asset and period
    for asset_name, ticker in assets.items():
        print(f"\nAnalyzing {asset_name} ({ticker}) across multiple time periods...")
        
        try:
            # Skip certain periods based on data availability
            asset_periods = periods.copy()
            if asset_name == 'Bitcoin':
                # Bitcoin data typically only available since ~2013
                asset_periods = [p for p in periods if p <= 15]
                
            if asset_name == 'Gold':
                # GLD ETF started in 2004
                asset_periods = [p for p in periods if p <= 20]
                
            all_results[asset_name] = calculate_multi_period_returns(
                ticker, 
                end_date, 
                asset_periods,
                initial_investment=10000,
                annual_investment=0
            )
            
        except Exception as e:
            print(f"Error analyzing {asset_name}: {e}")
    
    # Create comparison charts
    if all_results:
        # XIRR Comparison across time periods
        plt.figure(figsize=(15, 8))
        
        # Prepare data for plotting
        bar_width = 0.2
        
        # Set position of bars on X axis
        common_periods = [1, 5, 10]  # Periods likely available for all assets
        positions = np.arange(len(common_periods))
        
        # Colors for each asset
        colors = {
            'Gold': 'goldenrod',
            'S&P 500': 'darkblue',
            'Bitcoin': 'green'
        }
        
        # Plot XIRR for each asset and common period
        i = 0
        for asset_name in all_results:
            # Extract XIRR values for common periods
            xirr_values = []
            for period in common_periods:
                if period in all_results[asset_name]:
                    xirr_values.append(all_results[asset_name][period]['XIRR'])
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
        plt.savefig('xirr_comparison_periods.png')
        
        # Create heatmap for XIRR across all assets and periods
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        all_periods = sorted(list(set().union(*[all_results[asset].keys() for asset in all_results])))
        assets_list = list(all_results.keys())
        
        # Create data matrix for heatmap
        data_matrix = np.zeros((len(assets_list), len(all_periods)))
        
        for i, asset in enumerate(assets_list):
            for j, period in enumerate(all_periods):
                if period in all_results[asset]:
                    data_matrix[i, j] = all_results[asset][period]['XIRR']
                else:
                    data_matrix[i, j] = np.nan  # No data
        
        # Plot heatmap
        plt.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='XIRR (%)')
        
        # Set labels
        plt.yticks(np.arange(len(assets_list)), assets_list)
        plt.xticks(np.arange(len(all_periods)), all_periods)
        plt.xlabel('Investment Period (Years)')
        plt.title('XIRR Heatmap by Asset and Time Period')
        
        # Add text annotations
        for i in range(len(assets_list)):
            for j in range(len(all_periods)):
                value = data_matrix[i, j]
                if not np.isnan(value):
                    plt.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                             color='black' if 0 <= value < 20 else 'white')
        
        plt.tight_layout()
        plt.savefig('xirr_heatmap.png')
        
        # Line chart for XIRR trends over time
        plt.figure(figsize=(12, 6))
        
        for asset_name in all_results:
            periods = sorted(all_results[asset_name].keys())
            xirr_values = [all_results[asset_name][period]['XIRR'] for period in periods]
            
            plt.plot(periods, xirr_values, 'o-', label=asset_name, color=colors[asset_name])
            
            # Add values at each point
            for i, (period, xirr) in enumerate(zip(periods, xirr_values)):
                plt.text(period, xirr + 1, f'{xirr:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Investment Period (Years)')
        plt.ylabel('XIRR (%)')
        plt.title('XIRR Trends Over Different Time Periods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(sorted(list(set().union(*[all_results[asset].keys() for asset in all_results]))))
        
        plt.tight_layout()
        plt.savefig('xirr_trends.png')
        
        # Final value comparison (bar chart)
        plt.figure(figsize=(12, 6))
        
        i = 0
        for asset_name in all_results:
            # Extract final values for common periods
            final_values = []
            for period in common_periods:
                if period in all_results[asset_name]:
                    final_values.append(all_results[asset_name][period]['Final Value'])
                else:
                    final_values.append(0)  # No data
            
            # Plot bars
            bars = plt.bar(positions + i*bar_width, final_values, bar_width, label=asset_name, color=colors[asset_name])
            
            # Add values on top of bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:  # Only add text if there's data
                    plt.text(bar.get_x() + bar.get_width()/2., height + 500,
                            f'${height:.0f}', ha='center', va='bottom', fontsize=8)
            
            i += 1
        
        # Add labels and legend
        plt.xlabel('Investment Period (Years)')
        plt.ylabel('Final Value ($)')
        plt.title('Final Investment Value Comparison')
        plt.xticks(positions + bar_width, common_periods)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('final_value_comparison.png')
        
        # Print detailed results table
        print("\n" + "="*80)
        print("DETAILED RESULTS SUMMARY")
        print("="*80)
        
        for asset_name in all_results:
            print(f"\n{asset_name} Results:")
            print("-" * 80)
            print(f"{'Period (Years)':<15} {'XIRR':<15} {'Total Return':<15} {'Final Value':<15} {'Total Invested':<15}")
            print("-" * 80)
            
            periods = sorted(all_results[asset_name].keys())
            for period in periods:
                result = all_results[asset_name][period]
                print(f"{period:<15} {result['XIRR']:.2f}% {result['Total Return']:.2f}% ${result['Final Value']:.2f} ${result['Total Invested']:.2f}")
        
        plt.show()

if __name__ == "__main__":
    main()