import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import yfinance as yf
from datetime import datetime, timedelta
import traceback
def xirr(cashflows):
    """
    Calculate the Extended Internal Rate of Return (XIRR) for a series of cashflows.
    
    Parameters:
    cashflows: DataFrame with 'date' and 'amount' columns
    
    Returns:
    xirr_value: The annualized internal rate of return
    """
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
        except ValueError:
            # If brentq fails, try with other bounds or methods
            return optimize.brentq(lambda r: xnpv(r, cashflows), -0.9, 0.9)

def calculate_investment_return(ticker, start_date, end_date, initial_investment=10000, additional_investments=None):
    """
    Calculate the XIRR for an investment in a specific asset.
    
    Parameters:
    ticker: The ticker symbol for the asset
    start_date: The start date for the investment period
    end_date: The end date for the investment period
    initial_investment: The initial investment amount
    additional_investments: Optional DataFrame with 'date' and 'amount' columns for additional investments
    
    Returns:
    xirr_value: The annualized internal rate of return
    final_value: The final value of the investment
    cashflows: DataFrame with the cashflows used for the XIRR calculation
    """
    # Download historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Fixed: Check if data is empty properly
    if len(data) == 0:
        print(f"No data available for {ticker} in the specified date range.")
        return None, None, None
    
    # Create cashflows DataFrame
    cashflows = pd.DataFrame({'date': [start_date], 'amount': [-initial_investment]})
    
    # Add additional investments if provided
    if additional_investments is not None:
        for _, row in additional_investments.iterrows():
            cashflows = pd.concat([cashflows, pd.DataFrame({'date': [row['date']], 'amount': [-row['amount']]})])

    # Calculate final value
    initial_price = data['Close'].iloc[0]
    final_price = data['Close'].iloc[-1]
    
    # Initial shares
    shares = initial_investment / initial_price
    
    # Add additional shares from additional investments
    if additional_investments is not None:
        for _, row in additional_investments.iterrows():
            investment_date = row['date']
            # Find the closest trading day
            closest_date = data.index[data.index >= pd.Timestamp(investment_date)][0]
            price_on_date = data.loc[closest_date, 'Close']
            additional_shares = row['amount'] / price_on_date
            shares += additional_shares
    
    final_value = shares * final_price
    
    # Add final cashflow (selling the investment)
    cashflows = pd.concat([cashflows, pd.DataFrame({'date': [end_date], 'amount': [final_value[ticker]]})])
    
    # Calculate XIRR
    xirr_value = xirr(cashflows)
    
    return xirr_value, final_value, cashflows

def main():

    parser = argparse.ArgumentParser(description='XIRR calculator')
    parser.add_argument('--years', type=int, default=10, help='XIRR period in years (default: 10)')
    # parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    # parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    # parser.add_argument('--initial_investment', type=float, default=10000, help='Initial investment amount (default: 10000)')
    # parser.add_argument('--annual_investment', type=float, help='Annual investment amount (if any)')
    
    # Parse arguments
    args = parser.parse_args()    
    # Define investment period (defaults to 10 years)
    years = args.years
    end_date = datetime(2025, 4, 16)  
    start_date = end_date - timedelta(days=365*years)
    
    # Format dates as strings for yfinance
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # Define assets to analyze
    assets = {
        'Gold': 'GLD',       # SPDR Gold Shares ETF
        'S&P 500': 'SPY',    # SPDR S&P 500 ETF
        'Bitcoin': 'BTC-USD' # Bitcoin USD
    }
    
    results = {}
    
    # Calculate XIRR for each asset
    for asset_name, ticker in assets.items():
        print(f"\nCalculating XIRR for {asset_name} ({ticker})...")
        
        # Should we invest equal amount annually or you want to define your own?
        annual_investment = None
        additional_investments = None
        
        # Start from one year after the initial investment
        if annual_investment is not None:
    
            # Generate dates for annual investments
            annual_investment_dates = []

            start_date_ts = pd.Timestamp(start_date)
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

            print (additional_investments) 

        try:
            if annual_investment is not None:
                xirr_value, final_value, cashflows = calculate_investment_return(
                    ticker, 
                    start_date_str, 
                    end_date_str,
                    initial_investment=10000,
                    additional_investments=additional_investments
                )
            else:
                xirr_value, final_value, cashflows = calculate_investment_return(
                    ticker, 
                    start_date_str, 
                    end_date_str,
                    initial_investment=10000
                )
            
            if xirr_value is not None:
                results[asset_name] = {
                    'XIRR': xirr_value * 100,  # Convert to percentage
                    'Final Value': final_value[ticker],
                    'Initial Investment': 10000,
                    'Total Return': (final_value[ticker] / 10000 - 1) * 100  # Convert to percentage
                }
                
                print(f"XIRR for {asset_name}: {xirr_value * 100:.2f}%")
                print(f"Final Value: ${final_value[ticker]:.2f}")
                print(f"Total Return: {(final_value[ticker] / 10000 - 1) * 100:.2f}%")
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error calculating XIRR for {asset_name}: {e}")
    
    # Create comparison chart
    if results:
        plt.figure(figsize=(12, 8))
        
        # XIRR Comparison
        plt.subplot(1, 2, 1)
        assets_list = list(results.keys())
        xirr_values = [results[asset]['XIRR'] for asset in assets_list]
        
        plt.bar(assets_list, xirr_values, color=['goldenrod', 'darkblue', 'orange'])
        plt.title(str(years) + '-Year XIRR Comparison (annualized return)')
        plt.ylabel('XIRR (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, value in enumerate(xirr_values):
            plt.text(i, value + 0.5, f'{value:.2f}%', ha='center')
        
        # Total Return Comparison
        plt.subplot(1, 2, 2)
        total_returns = [results[asset]['Total Return'] for asset in assets_list]
        
        plt.bar(assets_list, total_returns, color=['goldenrod', 'darkblue', 'orange'])
        plt.title(str(years) + '-Year Total Return Comparison')
        plt.ylabel('Total Return (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, value in enumerate(total_returns):
            plt.text(i, value + 0.5, f'{value:.2f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig('investment_comparison.png')
        plt.show()
        
        print("\nSummary of Results:")
        print("-" * 60)
        print(f"{'Asset':<10} {'XIRR':<15} {'Total Return':<15} {'Final Value':<15}")
        print("-" * 60)
        
        for asset in assets_list:
            print(f"{asset:<10} {results[asset]['XIRR']:.2f}% {results[asset]['Total Return']:.2f}% ${results[asset]['Final Value']:.2f}")

# Alternative approach for Bitcoin data if yfinance has issues
def get_bitcoin_data_alternative(start_date, end_date):
    """
    Alternative method to get Bitcoin data if yfinance has issues.
    This function shows how to implement an alternative data source.
    """
    try:
        # Example using CoinGecko API (you would need to install the pycoingecko package)
        from pycoingecko import CoinGeckoAPI
        cg = CoinGeckoAPI()
        
        # Convert dates to Unix timestamps
        from_timestamp = int(pd.Timestamp(start_date).timestamp())
        to_timestamp = int(pd.Timestamp(end_date).timestamp())
        
        # Get historical data
        bitcoin_data = cg.get_coin_market_chart_range_by_id(
            id='bitcoin',
            vs_currency='usd',
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp
        )
        
        # Process the data into a DataFrame
        prices = bitcoin_data['prices']
        dates = [datetime.fromtimestamp(price[0]/1000) for price in prices]
        close_prices = [price[1] for price in prices]
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': close_prices,
            'Close': close_prices  # Using Close as Adj Close for consistency
        })
        df.set_index('Date', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Alternative Bitcoin data fetch failed: {e}")
        return None

if __name__ == "__main__":
    main()