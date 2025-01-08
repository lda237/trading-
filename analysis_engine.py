# src/analysis_engine.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

class TechnicalAnalyzer:
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(data, short=12, long=26, signal=9):
        short_ema = data['close'].ewm(span=short, adjust=False).mean()
        long_ema = data['close'].ewm(span=long, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band

    @staticmethod
    def calculate_ema(data, period):
        return data['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

class SignalType(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class TradingLevels:
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_ratio: float
    risk_amount: float
    potential_reward: float

@dataclass
class Signal:
    timestamp: pd.Timestamp
    type: SignalType
    symbol: str
    price: float
    confidence: float
    indicators: Dict[str, float]
    message: str

class SignalAnalyzer:
    def __init__(self):
        self.signals: List[Signal] = []
        self.technical_analyzer = TechnicalAnalyzer()


    def calculate_trading_levels(self, data: pd.DataFrame, signal_type: SignalType, 
                            current_price: float, atr_value: float) -> TradingLevels:
        """Calculate trading levels based on ATR and market structure"""
        
        # Use the provided ATR value instead of calculating it again
        latest_atr = atr_value
        
        # Définir les multiplicateurs pour SL et TP basés sur la volatilité
        sl_multiplier = 1.5
        tp1_multiplier = 2.0
        tp2_multiplier = 3.0
        tp3_multiplier = 4.0
        
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            stop_loss = current_price - (latest_atr * sl_multiplier)
            take_profit_1 = current_price + (latest_atr * tp1_multiplier)
            take_profit_2 = current_price + (latest_atr * tp2_multiplier)
            take_profit_3 = current_price + (latest_atr * tp3_multiplier)
        else:
            stop_loss = current_price + (latest_atr * sl_multiplier)
            take_profit_1 = current_price - (latest_atr * tp1_multiplier)
            take_profit_2 = current_price - (latest_atr * tp2_multiplier)
            take_profit_3 = current_price - (latest_atr * tp3_multiplier)
        
        # Calculer le ratio risque/récompense
        risk_amount = abs(current_price - stop_loss)
        potential_reward = abs(take_profit_2 - current_price)  # Using TP2 as reference
        risk_reward_ratio = potential_reward / risk_amount if risk_amount != 0 else 0
        
        return TradingLevels(
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            risk_reward_ratio=risk_reward_ratio,
            risk_amount=risk_amount,
            potential_reward=potential_reward
        )

    def identify_key_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identifier les niveaux clés du marché"""
        
        highs = data['high'].rolling(window=20).max()
        lows = data['low'].rolling(window=20).min()
        
        # Identifier les points pivot
        pivot_high = data['high'].iloc[-20:].max()
        pivot_low = data['low'].iloc[-20:].min()
        
        # Structure de prix récente
        recent_resistance = data['high'].iloc[-5:].max()
        recent_support = data['low'].iloc[-5:].min()
        
        return {
            'pivot_high': pivot_high,
            'pivot_low': pivot_low,
            'recent_resistance': recent_resistance,
            'recent_support': recent_support,
            'daily_high': data['high'].iloc[-1],
            'daily_low': data['low'].iloc[-1]
        }

    def analyze_trend_alignment(self, data: pd.DataFrame) -> str:
        """Analyser l'alignement des tendances sur différentes périodes"""
        
        ma20 = data['close'].rolling(window=20).mean()
        ma50 = data['close'].rolling(window=50).mean()
        ma200 = data['close'].rolling(window=200).mean()
        
        current_price = data['close'].iloc[-1]
        
        if current_price > ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]:
            return "Strong Uptrend"
        elif current_price < ma20.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]:
            return "Strong Downtrend"
        elif current_price > ma200.iloc[-1]:
            return "Bullish Bias"
        elif current_price < ma200.iloc[-1]:
            return "Bearish Bias"
        else:
            return "Mixed"

    def check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """Vérifier la confirmation du volume"""
        
        avg_volume = data['tick_volume'].rolling(window=20).mean()
        current_volume = data['tick_volume'].iloc[-1]
        
        return current_volume > avg_volume.iloc[-1] * 1.5
        
    def analyze_price_action(self, data: pd.DataFrame) -> List[dict]:
        """Analyze price action patterns"""
        patterns = []
        
        # Identify potential support/resistance levels
        highs = data['high'].rolling(window=20).max()
        lows = data['low'].rolling(window=20).min()
        
        # Detect price breaks
        price_breaks = []
        for i in range(3, len(data)):
            if (data['close'].iloc[i] > highs.iloc[i-1] and 
                data['tick_volume'].iloc[i] > data['tick_volume'].iloc[i-1]):
                price_breaks.append({"type": "breakout", "index": i})
            elif (data['close'].iloc[i] < lows.iloc[i-1] and 
                  data['tick_volume'].iloc[i] > data['tick_volume'].iloc[i-1]):
                price_breaks.append({"type": "breakdown", "index": i})
        
        return price_breaks

    def analyze_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze momentum indicators"""
        momentum = {}
        
        # RSI Analysis
        rsi = self.technical_analyzer.calculate_rsi(data)
        momentum['rsi'] = rsi.iloc[-1]
        momentum['rsi_trend'] = 'bullish' if rsi.iloc[-1] > rsi.iloc[-2] else 'bearish'
        
        # MACD Analysis
        macd, signal = self.technical_analyzer.calculate_macd(data)
        momentum['macd'] = macd.iloc[-1]
        momentum['macd_signal'] = signal.iloc[-1]
        momentum['macd_hist'] = macd.iloc[-1] - signal.iloc[-1]
        momentum['macd_trend'] = 'bullish' if momentum['macd_hist'] > 0 else 'bearish'
        
        # tick_volume Analysis
        momentum['volume_trend'] = 'up' if data['tick_volume'].iloc[-1] > data['tick_volume'].iloc[-2] else 'down'
        
        # Add volume strength indicator
        vol_sma = data['tick_volume'].rolling(window=20).mean()
        momentum['volume_strength'] = 'high' if data['tick_volume'].iloc[-1] > vol_sma.iloc[-1] * 1.5 else 'normal'
        
        return momentum


        # Add new methods for improved analysis
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    

    @staticmethod
    def calculate_volume_profile(data: pd.DataFrame, price_levels: int = 10) -> Dict[float, float]:
        """Calculate volume profile"""
        price_range = data['high'].max() - data['low'].min()
        price_step = price_range / price_levels
        
        volume_profile = {}
        for i in range(price_levels):
            price_level = data['low'].min() + (i * price_step)
            mask = (data['close'] >= price_level) & (data['close'] < price_level + price_step)
            volume_profile[price_level] = data.loc[mask, 'tick_volume'].sum()
            
        return volume_profile



    
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, str]:
        """Analyze overall trend"""
        trend = {}
        
        # Moving Averages
        ma20 = data['close'].rolling(window=20).mean()
        ma50 = data['close'].rolling(window=50).mean()
        ma200 = data['close'].rolling(window=200).mean()
        
        # Trend determination
        trend['short_term'] = 'bullish' if ma20.iloc[-1] > ma50.iloc[-1] else 'bearish'
        trend['long_term'] = 'bullish' if ma50.iloc[-1] > ma200.iloc[-1] else 'bearish'
        
        # Trend strength
        trend['strength'] = self._calculate_trend_strength(data, ma20, ma50, ma200)
        
        return trend

    def _calculate_trend_strength(self, data, ma20, ma50, ma200) -> str:
        strength = 0
        
        # Price above/below moving averages
        if data['close'].iloc[-1] > ma20.iloc[-1]: strength += 1
        if data['close'].iloc[-1] > ma50.iloc[-1]: strength += 1
        if data['close'].iloc[-1] > ma200.iloc[-1]: strength += 1
        
        # Moving average alignment
        if ma20.iloc[-1] > ma50.iloc[-1]: strength += 1
        if ma50.iloc[-1] > ma200.iloc[-1]: strength += 1
        
        if strength >= 4: return 'strong'
        elif strength >= 2: return 'moderate'
        else: return 'weak'

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """Generate trading signal based on comprehensive analysis"""
        
        # Get various analyses
        momentum = self.analyze_momentum(data)
        trend = self.analyze_trend(data)
        price_action = self.analyze_price_action(data)
        
        # Calculate Bollinger Bands
        upper_band, lower_band = self.technical_analyzer.calculate_bollinger_bands(data)
        current_price = data['close'].iloc[-1]
        
        # Initialize signal components
        signal_type = SignalType.NEUTRAL
        confidence = 0.0
        message_parts = []
        
        # Trend Analysis
        if trend['long_term'] == 'bullish' and trend['short_term'] == 'bullish':
            confidence += 0.3
            if trend['strength'] == 'strong':
                confidence += 0.2
                message_parts.append("Strong bullish trend")
        elif trend['long_term'] == 'bearish' and trend['short_term'] == 'bearish':
            confidence -= 0.3
            if trend['strength'] == 'strong':
                confidence -= 0.2
                message_parts.append("Strong bearish trend")
        
        # Momentum Analysis
        if momentum['rsi'] < 30 and momentum['macd_trend'] == 'bullish':
            confidence += 0.25
            message_parts.append("Oversold with positive momentum")
        elif momentum['rsi'] > 70 and momentum['macd_trend'] == 'bearish':
            confidence -= 0.25
            message_parts.append("Overbought with negative momentum")
        
        # tick_volume confirmation
        if momentum['volume_trend'] == 'up':
            confidence = confidence * 1.2
            message_parts.append("Strong tick_volume confirmation")
        
        # Recent price breaks
        recent_breaks = [b for b in price_action if b['index'] > len(data)-3]
        if recent_breaks:
            if recent_breaks[-1]['type'] == 'breakout':
                confidence += 0.15
                message_parts.append("Recent price breakout")
            else:
                confidence -= 0.15
                message_parts.append("Recent price breakdown")
        
        # Determine final signal type
        if confidence > 0.6:
            signal_type = SignalType.STRONG_BUY
        elif confidence > 0.2:
            signal_type = SignalType.BUY
        elif confidence < -0.6:
            signal_type = SignalType.STRONG_SELL
        elif confidence < -0.2:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Create signal object
        signal = Signal(
            timestamp=pd.Timestamp.now(),  # Using current timestamp instead of data index
            type=signal_type,
            symbol=symbol,
            price=current_price,
            confidence=abs(confidence),
            indicators={
                'rsi': momentum['rsi'],
                'macd': momentum['macd'],
                'trend_strength': trend['strength']
            },
            message=" | ".join(message_parts)
        )
        
        self.signals.append(signal)
        return signal

class SignalManager:
    def __init__(self, analyzer: SignalAnalyzer):
        self.analyzer = analyzer
        self.active_signals = []

    def update_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """Update and manage trading signals"""
        # Generate new signal
        new_signal = self.analyzer.generate_signal(data, symbol)
        
        # Update active signals list
        self.active_signals.append(new_signal)
        
        # Remove old signals (keep last 24 hours)
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=24)
        self.active_signals = [s for s in self.active_signals 
                             if s.timestamp > cutoff_time]
        
        return self.active_signals

    def get_signal_summary(self) -> str:
        """Get a summary of current trading signals"""
        if not self.active_signals:
            return "No active signals"
        
        latest_signal = self.active_signals[-1]
        return (f"Latest Signal: {latest_signal.type.value}\n"
                f"Confidence: {latest_signal.confidence:.2%}\n"
                f"Reason: {latest_signal.message}")