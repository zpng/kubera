"""
Technical Indicators Module
Calculates technical indicators using pandas-ta and stockstats
"""

import pandas as pd
import pandas_ta as ta
from stockstats import wrap
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for stock analysis.
    Supports: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more.
    """

    # Indicator descriptions for analysts
    INDICATOR_INFO = {
        "sma_50": "50-day Simple Moving Average - Medium-term trend indicator",
        "sma_200": "200-day Simple Moving Average - Long-term trend benchmark",
        "ema_10": "10-day Exponential Moving Average - Short-term momentum",
        "ema_20": "20-day Exponential Moving Average - Short-term trend",
        "rsi": "Relative Strength Index (14) - Overbought/oversold indicator (70/30 thresholds)",
        "macd": "MACD Line - Momentum indicator via EMA differences",
        "macd_signal": "MACD Signal Line - EMA smoothing of MACD",
        "macd_hist": "MACD Histogram - Gap between MACD and signal line",
        "bb_upper": "Bollinger Upper Band - 2 std dev above 20 SMA",
        "bb_middle": "Bollinger Middle Band - 20 SMA baseline",
        "bb_lower": "Bollinger Lower Band - 2 std dev below 20 SMA",
        "atr": "Average True Range (14) - Volatility measure",
        "adx": "Average Directional Index - Trend strength indicator",
        "stoch_k": "Stochastic %K - Fast stochastic oscillator",
        "stoch_d": "Stochastic %D - Slow stochastic oscillator (signal)",
        "obv": "On-Balance Volume - Volume-based momentum indicator",
        "cci": "Commodity Channel Index - Cyclical trend indicator",
    }

    def __init__(self):
        """Initialize Technical Indicators calculator."""
        logger.info("TechnicalIndicators initialized")

    def calculate_all(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate multiple technical indicators at once.

        Args:
            data: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            indicators: List of indicators to calculate (defaults to all)

        Returns:
            DataFrame with original data + indicator columns

        Example:
            >>> from src.data.market_data import get_stock_data
            >>> from src.data.indicators import TechnicalIndicators
            >>>
            >>> data = get_stock_data("AAPL", days_back=100)
            >>> calc = TechnicalIndicators()
            >>> data_with_indicators = calc.calculate_all(data)
            >>> print(data_with_indicators[['Close', 'sma_50', 'rsi']].tail())
        """
        if data.empty:
            logger.warning("Empty DataFrame provided")
            return data

        # Make a copy to avoid modifying original
        df = data.copy()

        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Default to calculating all indicators
        if indicators is None:
            indicators = [
                'sma_50', 'sma_200', 'ema_10', 'ema_20',
                'rsi', 'macd', 'bb', 'atr', 'adx', 'stoch', 'obv'
            ]

        # Calculate each indicator
        for indicator in indicators:
            try:
                if indicator == 'sma_50':
                    df['sma_50'] = ta.sma(df['Close'], length=50)
                elif indicator == 'sma_200':
                    df['sma_200'] = ta.sma(df['Close'], length=200)
                elif indicator == 'ema_10':
                    df['ema_10'] = ta.ema(df['Close'], length=10)
                elif indicator == 'ema_20':
                    df['ema_20'] = ta.ema(df['Close'], length=20)
                elif indicator == 'rsi':
                    df['rsi'] = ta.rsi(df['Close'], length=14)
                elif indicator == 'macd':
                    macd = ta.macd(df['Close'])
                    if macd is not None:
                        df['macd'] = macd['MACD_12_26_9']
                        df['macd_signal'] = macd['MACDs_12_26_9']
                        df['macd_hist'] = macd['MACDh_12_26_9']
                elif indicator == 'bb':
                    bb = ta.bbands(df['Close'], length=20, std=2)
                    if bb is not None:
                        df['bb_lower'] = bb['BBL_20_2.0']
                        df['bb_middle'] = bb['BBM_20_2.0']
                        df['bb_upper'] = bb['BBU_20_2.0']
                elif indicator == 'atr':
                    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                elif indicator == 'adx':
                    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
                    if adx is not None:
                        df['adx'] = adx['ADX_14']
                elif indicator == 'stoch':
                    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
                    if stoch is not None:
                        df['stoch_k'] = stoch['STOCHk_14_3_3']
                        df['stoch_d'] = stoch['STOCHd_14_3_3']
                elif indicator == 'obv':
                    df['obv'] = ta.obv(df['Close'], df['Volume'])
                elif indicator == 'cci':
                    df['cci'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

                logger.debug(f"Calculated {indicator}")

            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")

        return df

    def get_latest_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get the most recent values for specified indicators.

        Args:
            data: DataFrame with OHLCV data
            indicators: List of indicators to calculate

        Returns:
            Dictionary mapping indicator names to their latest values

        Example:
            >>> calc = TechnicalIndicators()
            >>> latest = calc.get_latest_indicators(data, ['rsi', 'macd'])
            >>> print(f"Current RSI: {latest['rsi']:.2f}")
        """
        df_with_indicators = self.calculate_all(data, indicators)

        latest = {}
        for col in df_with_indicators.columns:
            if col in self.INDICATOR_INFO or col in ['macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d']:
                value = df_with_indicators[col].iloc[-1]
                if pd.notna(value):
                    latest[col] = round(float(value), 2)

        return latest

    def get_indicator_summary(
        self,
        data: pd.DataFrame,
        lookback_days: int = 5
    ) -> Dict[str, Any]:
        """
        Get a comprehensive summary of technical indicators.

        Args:
            data: DataFrame with OHLCV data
            lookback_days: Number of recent days to analyze

        Returns:
            Dictionary with indicator analysis and signals

        Example:
            >>> calc = TechnicalIndicators()
            >>> summary = calc.get_indicator_summary(data)
            >>> print(summary['trend'])
            >>> print(summary['momentum'])
            >>> print(summary['signals'])
        """
        df = self.calculate_all(data)

        # Get latest values
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest

        summary = {
            "timestamp": datetime.now().isoformat(),
            "close_price": round(latest['Close'], 2),
            "price_change_pct": round(((latest['Close'] - previous['Close']) / previous['Close']) * 100, 2),
            "trend": {},
            "momentum": {},
            "volatility": {},
            "volume": {},
            "signals": []
        }

        # Trend analysis
        if 'sma_50' in df.columns and pd.notna(latest.get('sma_50')):
            summary['trend']['sma_50'] = round(latest['sma_50'], 2)
            if latest['Close'] > latest['sma_50']:
                summary['signals'].append("Price above 50-day SMA (Bullish)")
            else:
                summary['signals'].append("Price below 50-day SMA (Bearish)")

        if 'sma_200' in df.columns and pd.notna(latest.get('sma_200')):
            summary['trend']['sma_200'] = round(latest['sma_200'], 2)
            if latest['Close'] > latest['sma_200']:
                summary['signals'].append("Price above 200-day SMA (Long-term Bullish)")

        # Momentum analysis
        if 'rsi' in df.columns and pd.notna(latest.get('rsi')):
            rsi_val = latest['rsi']
            summary['momentum']['rsi'] = round(rsi_val, 2)
            if rsi_val > 70:
                summary['signals'].append(f"RSI Overbought ({rsi_val:.1f} > 70)")
            elif rsi_val < 30:
                summary['signals'].append(f"RSI Oversold ({rsi_val:.1f} < 30)")

        if 'macd' in df.columns and pd.notna(latest.get('macd')):
            summary['momentum']['macd'] = round(latest['macd'], 2)
            summary['momentum']['macd_signal'] = round(latest['macd_signal'], 2)
            summary['momentum']['macd_hist'] = round(latest['macd_hist'], 2)

            # MACD crossover signal
            if previous.get('macd') < previous.get('macd_signal') and latest['macd'] > latest['macd_signal']:
                summary['signals'].append("MACD Bullish Crossover")
            elif previous.get('macd') > previous.get('macd_signal') and latest['macd'] < latest['macd_signal']:
                summary['signals'].append("MACD Bearish Crossover")

        # Volatility analysis
        if 'atr' in df.columns and pd.notna(latest.get('atr')):
            summary['volatility']['atr'] = round(latest['atr'], 2)

        if 'bb_upper' in df.columns and pd.notna(latest.get('bb_upper')):
            summary['volatility']['bb_upper'] = round(latest['bb_upper'], 2)
            summary['volatility']['bb_middle'] = round(latest['bb_middle'], 2)
            summary['volatility']['bb_lower'] = round(latest['bb_lower'], 2)

            # Bollinger Band signals
            if latest['Close'] >= latest['bb_upper']:
                summary['signals'].append("Price at/above Upper Bollinger Band (Overbought)")
            elif latest['Close'] <= latest['bb_lower']:
                summary['signals'].append("Price at/below Lower Bollinger Band (Oversold)")

        # Volume analysis
        if 'obv' in df.columns and pd.notna(latest.get('obv')):
            summary['volume']['obv'] = int(latest['obv'])

        logger.info(f"Generated indicator summary with {len(summary['signals'])} signals")
        return summary


def get_indicator_values(symbol: str, indicators: List[str], days_back: int = 100) -> Dict[str, float]:
    """
    Convenience function to get indicator values for a symbol.

    Args:
        symbol: Stock ticker symbol
        indicators: List of indicators to calculate
        days_back: Number of days of historical data

    Returns:
        Dictionary with latest indicator values
    """
    from .market_data import get_stock_data

    data = get_stock_data(symbol, days_back=days_back)
    calc = TechnicalIndicators()
    return calc.get_latest_indicators(data, indicators)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    from market_data import get_stock_data

    print("\n=== Testing Technical Indicators ===")

    # Get stock data
    data = get_stock_data("AAPL", days_back=100)
    print(f"Retrieved {len(data)} days of data for AAPL")

    # Calculate indicators
    calc = TechnicalIndicators()
    data_with_indicators = calc.calculate_all(data)

    # Show latest values
    print("\n=== Latest Indicator Values ===")
    latest = calc.get_latest_indicators(data)
    for indicator, value in latest.items():
        if indicator in calc.INDICATOR_INFO:
            print(f"{indicator.upper()}: {value:.2f} - {calc.INDICATOR_INFO[indicator]}")

    # Get comprehensive summary
    print("\n=== Indicator Summary ===")
    summary = calc.get_indicator_summary(data)
    print(f"\nPrice: ${summary['close_price']} ({summary['price_change_pct']:+.2f}%)")
    print(f"\nSignals ({len(summary['signals'])} total):")
    for signal in summary['signals']:
        print(f"  - {signal}")

    print(f"\nTrend: {summary['trend']}")
    print(f"Momentum: {summary['momentum']}")
