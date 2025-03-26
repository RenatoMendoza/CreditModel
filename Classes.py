import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm

class Get_Historical_Data:
    """
    A class to fetch historical stock price and financial statements using Yahoo Finance.

    Attributes:
    - ticker (str): Stock ticker symbol.
    - start_date (str): Start date for historical price data (YYYY-MM-DD).
    - end_date (str): End date for historical price data (YYYY-MM-DD).
    """

    def __init__(self, ticker: str, start_date: str, end_date: str):
        """
        Initializes the Get_Historical_Data class.

        Parameters:
        - ticker (str): The stock ticker symbol.
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        """
        self.ticker = ticker
        self.start = datetime.datetime.strptime(self.start, '%Y-%m-%d')
        self.end = datetime.datetime.strptime(self.end, '%Y-%m-%d')
        self.balance = yf.Ticker(ticker).balance_sheet.iloc[:, :-1]
        self.estado = yf.Ticker(ticker).income_stmt.iloc[:, :-1]
        self.cashflow = yf.Ticker(ticker).cash_flow.iloc[:, :-1]


    def get_data(self) -> pd.Series:
        """
        Fetches historical closing price data for the given ticker.

        Returns:
        - pd.Series: Series containing historical closing prices.
        """
        try:
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)['Close']
            return data
        except Exception as e:
            print(f"Error fetching historical data for {self.ticker}: {e}")
            return None

    @staticmethod
    def get_historical_financials(ticker: str):
        """
        Fetches historical financial statements (balance sheet & income statement) for a given ticker.

        Parameters:
        - ticker (str): The stock ticker symbol.

        Returns:
        - tuple (pd.DataFrame, pd.DataFrame): Balance sheet and income statement as DataFrames.
        """
        try:
            yf_ticker = yf.Ticker(ticker)

            # Fetch balance sheet
            balance = yf_ticker.balance_sheet
            balance_df = pd.DataFrame(balance)

            # Fetch income statement
            income_stmt = yf_ticker.financials
            income_stmt_df = pd.DataFrame(income_stmt)

            return balance_df, income_stmt_df
        except Exception as e:
            print(f"Error fetching financial data for {ticker}: {e}")
            return None, None
        
class Credit_Models:
    """
    A class containing credit risk models: 
    - Merton Black-Scholes Model for Probability of Default (PD)
    - Altman Z-score for financial distress prediction
    """
    @staticmethod
    def Merton_BS_Model(balance, stock_price_per_share, sigma, rf, T):
        """
        Calculates the Probability of Default (PD) using the Merton Model.

        Parameters:
        - balance: DataFrame containing balance sheet information.
        - stock_price_per_share: Current stock price.
        - sigma: Volatility of the company's assets.
        - rf: Risk-free rate.
        - T: Time horizon in years.

        Returns:
        - Probability of Default (PD).
        """
        try:
        
            # Market Capitalization (Equity Value)
            market_capitalization = balance.loc['Ordinary Shares Number'].iloc[0] * stock_price_per_share

            # Total Debt (Liabilities)
            D = balance.loc['Total Liabilities Net Minority Interest'].iloc[0]

            # Market Value of Assets
            V = market_capitalization + D

            # Distance to Default (DD)
            DD = norm.cdf(np.log(V / D) + (rf - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

            # Probability of Default (PD)
            PD = 1 - norm.cdf(DD)

            return PD


        except Exception as e:
            print(f"Error in Merton_BS_Model: {e}")
            return None

    @staticmethod
    def Altman_Zscore(balance, incomestmt, stock_price_per_share):
        """
        Calculates the Altman Z-score using the correct column mappings.
        
        Parameters:
        - balance: DataFrame containing balance sheet information.
        - incomestmt: DataFrame containing income statement information.
        - stock_price_per_share: float, stock price per share (Required for Market Value of Equity).
        
        Returns:
        - Z-score as a float value.
        """
        try:
            # X1: Working Capital / Total Assets
            X1 = balance.loc['Working Capital'].iloc[0] / balance.loc['Total Assets'].iloc[0]

            # X2: Retained Earnings / Total Assets
            X2 = balance.loc['Retained Earnings'].iloc[0] / balance.loc['Total Assets'].iloc[0]

            # X3: EBIT / Total Assets
            X3 = incomestmt.loc['EBIT'].iloc[0] / balance.loc['Total Assets'].iloc[0]

            # X4: Market Value of Equity / Total Liabilities
            market_value_equity = balance.loc['Ordinary Shares Number'].iloc[0] * stock_price_per_share
            X4 = market_value_equity / balance.loc['Total Liabilities Net Minority Interest'].iloc[0]
            
            # Altman Z-score formula
            Z = 6.56 * X1 + 3.26 * X2 + 6.72 * X3 + 1.05 * X4

            return Z

        except Exception as e:
            print(f"Error in Altman_Zscore: {e}")
            return None
        
    @staticmethod
    def credit_decision(Z_score, PD):
        """
        Determines whether a company should be approved or denied for credit 
        based on Altman Z-score and Probability of Default (PD).

        Parameters:
        - Z_score (float): Altman Z-score of the company.
        - PD (float): Probability of Default (PD) from the Merton Model (as a decimal).

        Returns:
        - str: "Approve", "Needs Further Review", or "Deny".
    """

    # Define decision based on Z-score and PD
        if Z_score >= 2.99 and PD <= 0.05:
            return "Approve ✅"

        elif (1.81 <= Z_score < 2.99) or (0.05 < PD <= 0.10):
            return "Needs Further Review ⚠️"

        else:
            return "Deny ❌"