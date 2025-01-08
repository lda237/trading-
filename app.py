import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from plyer import notification
from data_manager import DataConnector, DataValidator, DataCache
import MetaTrader5 as mt5
from analysis_engine import SignalAnalyzer, SignalManager, SignalType, TechnicalAnalyzer
import time
import json
from chart_components import ChartAnnotator, TDI, ChartAnnotation
from data_manager import DataConnector, DataError, DataValidator, DataCache

class Config:
    def __init__(self):
        self.load_config()

    def load_config(self):
        try:
            with open('user_config.json', 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            self.settings = {
                'theme': 'dark',
                'default_symbol': 'EURUSD',
                'default_timeframe': 'H1',
                'chart_style': 'candles',
                'indicators': ['RSI', 'MACD', 'BB'],
                'alert_sound': True,
                'email_alerts': False,
                'tdi_settings': {  # Ajouté
                    'rsi_period': 13,
                    'price_period': 2,
                    'bank_period': 34,
                    'volatility_period': 34
                }
            }
            self.save_config()

    def save_config(self):
        with open('user_config.json', 'w') as f:
            json.dump(self.settings, f)

class AlertManager:
    def __init__(self):
        self.alerts = []
        self.history = []
        self.max_history = 100

    def add_alert(self, message, level="info", timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        alert = {
            "message": message,
            "level": level,
            "timestamp": timestamp,
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        self.history.append(alert)
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self._send_push_notification(message)

    def _send_push_notification(self, message):
        try:
            notification.notify(
                title="Trading Alert",
                message=message,
                timeout=5,
                app_icon="assets/icon.ico"  # Custom icon path
            )
        except Exception as e:
            st.error(f"Notification error: {e}")

    def display_alerts(self):
        with st.expander("Active Alerts", expanded=True):
            for alert in reversed(self.alerts):
                self._display_alert_message(alert)

    def _display_alert_message(self, alert):
        timestamp = alert["timestamp"].strftime("%H:%M:%S")
        message = f"{timestamp} - {alert['message']}"
        
        if alert["level"] == "info":
            st.info(message)
        elif alert["level"] == "warning":
            st.warning(message)
        elif alert["level"] == "error":
            st.error(message)

class ChartManager:
    def __init__(self):
        self.timeframe_mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

    def create_chart(self, data, symbol, indicators):
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=5, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],  # Ajusté pour le TDI
            subplot_titles=('Price', 'Volume', 'RSI', 'MACD', 'TDI')

        )


        # Créer l'annotateur
        annotator = ChartAnnotator()
        
        # Ajouter les niveaux de support/résistance
        annotator.add_support_resistance(data)
        
        # Ajouter les signaux de trading s'ils existent
        if hasattr(self, 'signal_manager') and self.signal_manager.active_signals:
            for signal in self.signal_manager.active_signals:
                annotator.add_trading_signal(
                    signal.timestamp,
                    signal.price,
                    signal.type.value
                )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index if isinstance(data.index, pd.DatetimeIndex) else data['time'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=symbol
            ),
            row=1, col=1
        )

        # Add Volume
        colors = ['red' if c < o else 'green' for c, o in zip(data['close'], data['open'])]
        fig.add_trace(
            go.Bar(
                x=data.index if isinstance(data.index, pd.DatetimeIndex) else data['time'],
                y=data['tick_volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )


        # Add ATR if enabled
        if 'ATR' in indicators:
            atr = TechnicalAnalyzer.calculate_atr(data)
            fig.add_trace(
                go.Scatter(
                    x=data.index if isinstance(data.index, pd.DatetimeIndex) else data['time'],
                    y=atr,
                    name='ATR',
                    line=dict(color='purple')
                ),
                row=1, col=1,
                secondary_y=True
            )

        # Add Bollinger Bands if enabled
        if 'BB' in indicators:
            upper_band, lower_band = TechnicalAnalyzer.calculate_bollinger_bands(data)
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=upper_band,
                    name='Upper BB',
                    line=dict(color='rgba(250, 250, 250, 0.4)'),
                    fill=None
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=lower_band,
                    name='Lower BB',
                    line=dict(color='rgba(250, 250, 250, 0.4)'),
                    fill='tonexty'
                ),
                row=1, col=1
            )

        # Add RSI
        if 'RSI' in indicators:
            rsi = TechnicalAnalyzer.calculate_rsi(data)
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=rsi,
                    name='RSI',
                    line=dict(color='#17BECF')
                ),
                row=2, col=1
            )
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Add MACD
        if 'MACD' in indicators:
            macd, signal = TechnicalAnalyzer.calculate_macd(data)
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=macd,
                    name='MACD',
                    line=dict(color='#17BECF')
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=signal,
                    name='Signal',
                    line=dict(color='#FFA500')
                ),
                row=3, col=1
            )


        # Ajouter le TDI si activé
        if 'TDI' in indicators:
            tdi = TDI()
            fig = tdi.add_to_chart(fig, data)
        
        # Appliquer les annotations
        fig = annotator.apply_annotations(fig)


        # Update layout with improved styling
        fig.update_layout(
            height=900,  # Increased height
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"  # Semi-transparent background
            ),
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=25, b=10)
        )

        # Update axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)

        return fig

class DashboardUI:
    def __init__(self):
        self.config = Config()
        self.data_connector = DataConnector()
        self.data_validator = DataValidator()
        self.data_cache = DataCache()
        self.alert_manager = AlertManager()
        self.chart_manager = ChartManager()
        self.signal_analyzer = SignalAnalyzer()
        self.signal_manager = SignalManager(self.signal_analyzer)
        self.selected_symbol = self.config.settings['default_symbol']
        self.selected_timeframe = self.config.settings['default_timeframe']
        self.num_candles = 100

    def initialize_data(self):
        try:
            # Get available symbols
            symbols = self.data_connector.get_symbols()
            st.session_state.symbols_list = symbols
            
            # Get market data
            data = self.data_connector.get_data(
                symbol=self.selected_symbol,
                timeframe=self.selected_timeframe,
                num_candles=self.num_candles
            )
            
            return data
            
        except DataError as e:
            st.error(f"Data error: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return None

    def display_signal_panel(self, signal, symbol):
        # Create a styled container for the signal
        signal_color = {
            SignalType.STRONG_BUY: "success",
            SignalType.BUY: "info",
            SignalType.NEUTRAL: "warning",
            SignalType.SELL: "warning",
            SignalType.STRONG_SELL: "error"
        }

        with st.container():
            # Signal header with color coding
            st.markdown(f"""
                <div style='padding: 10px; background-color: {'#1a936f' if 'BUY' in signal.type.value else '#ff4b4b' if 'SELL' in signal.type.value else '#ffd166'}; 
                border-radius: 5px; margin-bottom: 10px;'>
                    <h3 style='color: white; margin: 0;'>{signal.type.value} - {symbol}</h3>
                </div>
                """, unsafe_allow_html=True)

            # Signal details in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", f"{signal.price:.5f}")
                st.metric("Confidence", f"{signal.confidence:.1%}")

            with col2:
                st.metric("RSI", f"{signal.indicators['rsi']:.1f}")
                st.metric("MACD", f"{signal.indicators['macd']:.5f}")

            with col3:
                st.metric("Trend Strength", signal.indicators['trend_strength'])

            # Calculate trading levels using data from SignalAnalyzer
            try:
                # Get the data for the current symbol
                data = self.data_cache.get_cached_data(symbol, self.selected_timeframe, self.num_candles)
                if data is not None:
                    trading_levels = self.signal_analyzer.calculate_trading_levels(
                        data=data,
                        signal_type=signal.type,
                        current_price=signal.price,
                        atr_value=self.signal_analyzer.calculate_atr(data).iloc[-1]
                    )

                    # Display trading levels
                    st.subheader("Trading Levels")
                    levels_col1, levels_col2 = st.columns(2)
                    
                    with levels_col1:
                        st.metric("Stop Loss", f"{trading_levels.stop_loss:.5f}", 
                                delta=f"{((trading_levels.stop_loss - signal.price)/signal.price)*100:.2f}%")
                        st.metric("Take Profit 1", f"{trading_levels.take_profit_1:.5f}",
                                delta=f"{((trading_levels.take_profit_1 - signal.price)/signal.price)*100:.2f}%")
                        st.metric("Take Profit 2", f"{trading_levels.take_profit_2:.5f}",
                                delta=f"{((trading_levels.take_profit_2 - signal.price)/signal.price)*100:.2f}%")
                    
                    with levels_col2:
                        st.metric("Risk/Reward", f"{trading_levels.risk_reward_ratio:.2f}")
                        st.metric("Risk Amount", f"{trading_levels.risk_amount:.5f}")
                        st.metric("Potential Reward", f"{trading_levels.potential_reward:.5f}")

            except Exception as e:
                st.warning(f"Could not calculate trading levels: {str(e)}")

            # Signal message
            st.info(signal.message)

    def initialize_session_state(self):
        if 'symbols_list' not in st.session_state:
            st.session_state.symbols_list = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "XAUUSD", "EURJPY"]
        if 'selected_indicators' not in st.session_state:
            st.session_state.selected_indicators = self.config.settings['indicators']
        if 'active_signals' not in st.session_state:
            st.session_state.active_signals = []
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = self.config.settings['default_symbol']
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = self.config.settings['default_timeframe']
        if 'num_candles' not in st.session_state:
            st.session_state.num_candles = 100

    def display_signals_history(self, signals):
        with st.expander("Signals History", expanded=True):
            for signal in reversed(signals[-5:]):  # Show last 5 signals
                timestamp = signal.timestamp.strftime("%H:%M:%S")
                st.markdown(f"""
                    <div style='padding: 5px; margin: 5px 0; border-left: 3px solid 
                    {'#1a936f' if 'BUY' in signal.type.value else '#ff4b4b' if 'SELL' in signal.type.value else '#ffd166'};'>
                        <small>{timestamp}</small><br>
                        <strong>{signal.type.value}</strong> @ {signal.price:.5f}
                    </div>
                    """, unsafe_allow_html=True)
                
    def create_sidebar(self):
        with st.sidebar:
            st.image("assets/logo.png", width=100)  # Add your logo
            st.title("Trading Analysis")

            # Symbol selection with search
            symbol = st.selectbox(
                "Select Symbol",
                st.session_state.symbols_list,
                index=st.session_state.symbols_list.index(self.config.settings['default_symbol'])
            )

            # Timeframe selection
            timeframe = st.selectbox(
                "Timeframe",
                list(self.chart_manager.timeframe_mapping.keys()),
                index=list(self.chart_manager.timeframe_mapping.keys()).index(
                    self.config.settings['default_timeframe']
                )
            )

            # Chart settings
            st.subheader("Chart Settings")
            indicators = st.multiselect(
                "Indicators",
                ["RSI", "MACD", "BB", "TDI"],
                default=st.session_state.selected_indicators
            )

            # Number of candles slider
            num_candles = st.slider(
                "Number of Candles",
                min_value=50,
                max_value=500,
                value=100,
                step=50
            )

            return symbol, timeframe, indicators, num_candles

    def run(self):
        st.set_page_config(page_title="Trading Analysis", layout="wide")
        self.initialize_session_state()

        # Create sidebar and get parameters
        symbol, timeframe, indicators, num_candles = self.create_sidebar()

        # Main content area
        col1, col2 = st.columns([7, 3])

        with col1:
            try:
                # Fetch and process data
                self.data_connector.connect()
                data = self.data_connector.get_data(
                    symbol,
                    self.chart_manager.timeframe_mapping[timeframe],
                    num_candles
                )
                self.data_connector.disconnect()
                
                # Validate data
                validated_data = DataValidator.validate_data(data)
                
                # Update cache with all required parameters
                self.data_cache.update_cache(
                    symbol=symbol,
                    timeframe=timeframe,
                    num_candles=num_candles,
                    data=validated_data  # Ajout du paramètre manquant
                )

                # Generate trading signals
                active_signals = self.signal_manager.update_signals(validated_data, symbol)
                st.session_state.active_signals = active_signals

                # Create and display chart
                fig = self.chart_manager.create_chart(validated_data, symbol, indicators)
                st.plotly_chart(fig, use_container_width=True)

                # Display quick stats with signals influence
                latest_signal = active_signals[-1] if active_signals else None
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.metric(
                        "Current Price",
                        f"{data['close'].iloc[-1]:.5f}",
                        delta=f"{((data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2] * 100):.2f}%"
                    )
                with stats_cols[1]:
                    st.metric(
                        "Signal",
                        latest_signal.type.value if latest_signal else "NO SIGNAL",
                        delta=f"{latest_signal.confidence:.1%}" if latest_signal else None
                    )
                with stats_cols[2]:
                    st.metric("24h High", f"{data['high'].max():.5f}")
                with stats_cols[3]:
                    st.metric("24h Low", f"{data['low'].min():.5f}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

        with col2:
            # Latest Signal Panel
            st.subheader("Current Signal")
            if st.session_state.active_signals:
                latest_signal = st.session_state.active_signals[-1]
                self.display_signal_panel(latest_signal, symbol)
            else:
                st.info("No active signals")

            # Signals History
            if st.session_state.active_signals:
                self.display_signals_history(st.session_state.active_signals)

            # Alerts panel
            st.subheader("Technical Alerts")
            self.alert_manager.display_alerts()

        # Trading panel
        with st.expander("Quick Trade", expanded=False):
            if latest_signal:
                signal_text = f"Current Signal: {latest_signal.type.value}"
                if "BUY" in latest_signal.type.value:
                    st.success(signal_text)
                elif "SELL" in latest_signal.type.value:
                    st.error(signal_text)
                else:
                    st.warning(signal_text)

                # Get trading levels
                try:
                    trading_levels = self.signal_analyzer.calculate_trading_levels(
                        data=validated_data,
                        signal_type=latest_signal.type,
                        current_price=latest_signal.price,
                        atr_value=self.signal_analyzer.calculate_atr(validated_data).iloc[-1]
                    )

                    # Pre-fill form with calculated levels
                    lot_size = st.number_input(
                        "Lot Size",
                        min_value=0.01,
                        max_value=10.0,
                        value=0.1,
                        step=0.01,
                        key="trade_lot_size"
                    )
                    sl_price = st.number_input(
                        "Stop Loss Price", 
                        value=float(trading_levels.stop_loss),
                        format="%.5f",
                        key="trade_sl_price"
                    )
                    tp1_price = st.number_input(
                        "Take Profit 1 Price", 
                        value=float(trading_levels.take_profit_1),
                        format="%.5f",
                        key="trade_tp1_price"
                    )
                    tp2_price = st.number_input(
                        "Take Profit 2 Price", 
                        value=float(trading_levels.take_profit_2),
                        format="%.5f",
                        key="trade_tp2_price"
                    )

                    # Display risk metrics
                    risk_cols = st.columns(2)
                    with risk_cols[0]:
                        st.metric("Risk/Reward Ratio", f"{trading_levels.risk_reward_ratio:.2f}")
                    with risk_cols[1]:
                        st.metric("Risk Amount", f"{trading_levels.risk_amount:.2f}")

                except Exception as e:
                    st.warning(f"Could not calculate trading levels: {str(e)}")
                    # Fallback to basic form
                    lot_size = st.number_input(
                        "Lot Size",
                        min_value=0.01,
                        max_value=10.0,
                        value=0.1,
                        step=0.01,
                        key="fallback_lot_size"
                    )
                    sl_points = st.number_input(
                        "Stop Loss (points)",
                        min_value=1,
                        max_value=500,
                        value=50,
                        key="fallback_sl_points"
                    )
                    tp_points = st.number_input(
                        "Take Profit (points)",
                        min_value=1,
                        max_value=1000,
                        value=100,
                        key="fallback_tp_points"
                    )

            # Trading buttons
            cols = st.columns(2)
            with cols[0]:
                buy_button = st.button(
                    "Buy",
                    type="primary",
                    disabled=latest_signal and "SELL" in latest_signal.type.value,
                    key="buy_button"
                )
            with cols[1]:
                sell_button = st.button(
                    "Sell",
                    type="secondary",
                    disabled=latest_signal and "BUY" in latest_signal.type.value,
                    key="sell_button"
                )
    

if __name__ == "__main__":
    dashboard = DashboardUI()
    dashboard.run()