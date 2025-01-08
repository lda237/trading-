# chart_components.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import plotly.graph_objects as go

@dataclass
class ChartAnnotation:
    x: pd.Timestamp
    y: float
    text: str
    color: str = "white"
    arrow_color: str = "yellow"
    font_size: int = 12
    arrow_size: int = 2

class ChartAnnotator:
    def __init__(self):
        self.annotations = []
        self.support_levels = []
        self.resistance_levels = []
        
    def add_trading_signal(self, timestamp: pd.Timestamp, price: float, signal_type: str):
        """Add trading signal annotation"""
        color = "green" if signal_type.upper() == "BUY" else "red"
        arrow = "↑" if signal_type.upper() == "BUY" else "↓"
        self.annotations.append(
            ChartAnnotation(
                x=timestamp,
                y=price,
                text=f"{arrow} {signal_type}",
                color=color
            )
        )
        
    def add_support_resistance(self, data: pd.DataFrame, window: int = 20):
        """Identify and add support/resistance levels"""
        highs = data['high'].rolling(window=window).max()
        lows = data['low'].rolling(window=window).min()
        
        # Find significant levels
        for i in range(window, len(data)):
            # Support levels
            if (lows.iloc[i] == lows.iloc[i-window:i+1].min() and
                data['tick_volume'].iloc[i] > data['tick_volume'].iloc[i-window:i+1].mean()):
                self.support_levels.append(lows.iloc[i])
                
            # Resistance levels    
            if (highs.iloc[i] == highs.iloc[i-window:i+1].max() and
                data['tick_volume'].iloc[i] > data['tick_volume'].iloc[i-window:i+1].mean()):
                self.resistance_levels.append(highs.iloc[i])
                
    def add_custom_annotation(self, annotation: ChartAnnotation):
        """Add custom annotation"""
        self.annotations.append(annotation)
        
    def apply_annotations(self, fig: go.Figure) -> go.Figure:
        """Apply all annotations to the figure"""
        # Add support/resistance lines
        for level in self.support_levels:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                row=1, col=1
            )
            
        for level in self.resistance_levels:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=1, col=1
            )
            
        # Add custom annotations
        for ann in self.annotations:
            fig.add_annotation(
                x=ann.x,
                y=ann.y,
                text=ann.text,
                showarrow=True,
                font=dict(size=ann.font_size, color=ann.color),
                arrowcolor=ann.arrow_color,
                arrowsize=ann.arrow_size,
                row=1, col=1
            )
            
        return fig

class TDI:
    """Traders Dynamic Index Implementation"""
    
    def __init__(self, rsi_period: int = 13, price_period: int = 2,
                 bank_period: int = 34, volatility_period: int = 34):
        self.rsi_period = rsi_period
        self.price_period = price_period
        self.bank_period = bank_period
        self.volatility_period = volatility_period
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate TDI components"""
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Market Base Line (MBL)
        mbl = rsi.rolling(window=self.bank_period).mean()
        
        # Calculate Volatility Bands
        std = rsi.rolling(window=self.volatility_period).std()
        upper_band = mbl + std
        lower_band = mbl - std
        
        # Calculate Signal Line
        signal = rsi.rolling(window=self.price_period).mean()
        
        return {
            'rsi': rsi,
            'signal': signal,
            'mbl': mbl,
            'upper_band': upper_band,
            'lower_band': lower_band
        }
        
    def add_to_chart(self, fig: go.Figure, data: pd.DataFrame, row: int = 4) -> go.Figure:
        """Add TDI to the chart"""
        tdi_data = self.calculate(data)
        
        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=tdi_data['rsi'],
                name='TDI RSI',
                line=dict(color='yellow', width=1),
            ),
            row=row, col=1
        )
        
        # Add Signal line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=tdi_data['signal'],
                name='TDI Signal',
                line=dict(color='blue', width=1),
            ),
            row=row, col=1
        )
        
        # Add Market Base Line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=tdi_data['mbl'],
                name='Market Base Line',
                line=dict(color='green', width=1),
            ),
            row=row, col=1
        )
        
        # Add Volatility Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=tdi_data['upper_band'],
                name='Upper Band',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=tdi_data['lower_band'],
                name='Lower Band',
                line=dict(color='red', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
            ),
            row=row, col=1
        )
        
        # Update y-axis
        fig.update_yaxes(title_text="TDI", row=row)
        
        return fig