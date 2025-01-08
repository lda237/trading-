# src/data_manager.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union
import json
import time
from pathlib import Path
import logging
from functools import lru_cache

# data_manager.py

import MetaTrader5 as mt5
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from functools import lru_cache
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataError(Exception):
    """Base class for data-related errors"""
    pass

class ConnectionError(DataError):
    """Error raised when connection to data source fails"""
    pass

class DataFetchError(DataError):
    """Error raised when data fetching fails"""
    pass

class ValidationError(DataError):
    """Error raised when data validation fails"""
    pass

@dataclass
class MarketData:
    """Data structure for market data"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    last_update: datetime
    metadata: Dict = None

class DataValidator:
    """Validates market data integrity and format"""
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates market data for integrity and correct format
        
        Args:
            data: DataFrame containing market data
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValidationError: If data fails validation
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'time']
            
            # Check required columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")

            # Check for NaN values
            if data.isnull().any().any():
                logger.warning(f"Found NaN values in data. Rows before cleanup: {len(data)}")
                data = data.dropna()
                logger.info(f"Rows after NaN cleanup: {len(data)}")
                if len(data) == 0:
                    raise ValidationError("No valid data after removing NaN values")

            # Check for negative prices
            price_cols = ['open', 'high', 'low', 'close']
            if (data[price_cols] < 0).any().any():
                raise ValidationError("Negative prices detected in data")

            # Check price relationships
            if (data['high'] < data['low']).any():
                raise ValidationError("High price less than low price detected")
            if ((data['close'] > data['high']) | (data['close'] < data['low'])).any():
                raise ValidationError("Close price outside high-low range")
            if ((data['open'] > data['high']) | (data['open'] < data['low'])).any():
                raise ValidationError("Open price outside high-low range")

            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(data['time']):
                data['time'] = pd.to_datetime(data['time'])

            # Sort by time
            data = data.sort_values('time')

            # Remove duplicates
            data = data.drop_duplicates(subset=['time'])

            return data

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise ValidationError(f"Data validation failed: {str(e)}")

class DataCache:
    """Manages caching of market data"""
    
    def __init__(self, cache_duration: timedelta = timedelta(hours=1)):
        self.cache: Dict[str, MarketData] = {}
        self.cache_duration = cache_duration
        self._load_cache()

    def _load_cache(self):
        """Load cached data from disk if available"""
        cache_file = Path('market_data_cache.pkl')
        if cache_file.exists():
            try:
                self.cache = pd.read_pickle(cache_file)
                logger.info("Cache loaded from disk successfully")
            except Exception as e:
                logger.warning(f"Failed to load cache from disk: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            pd.to_pickle(self.cache, 'market_data_cache.pkl')
            logger.info("Cache saved to disk successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def get_cached_data(self, symbol: str, timeframe: str, num_candles: int) -> Optional[pd.DataFrame]:
        """Retrieve data from cache if available and not expired"""
        cache_key = f"{symbol}_{timeframe}_{num_candles}"
        
        if cache_key in self.cache:
            market_data = self.cache[cache_key]
            if datetime.now() - market_data.last_update < self.cache_duration:
                logger.info(f"Cache hit for {cache_key}")
                return market_data.data
            else:
                logger.info(f"Cache expired for {cache_key}")
                del self.cache[cache_key]
        
        return None

    def update_cache(self, symbol: str, timeframe: str, num_candles: int, data: pd.DataFrame):
        """Update cache with new market data"""
        cache_key = f"{symbol}_{timeframe}_{num_candles}"
        self.cache[cache_key] = MarketData(
            symbol=symbol,
            timeframe=timeframe,
            data=data,
            last_update=datetime.now()
        )
        self._save_cache()

class DataConnector:
    """Manages connection to MT5 and data retrieval"""
    
    def __init__(self):
        self.cache = DataCache()
        self.connected = False
        self.retry_attempts = 3
        self.retry_delay = 1
        self.validator = DataValidator()

    @property
    def timeframe_mapping(self) -> Dict[str, int]:
        """Map string timeframes to MT5 constants and vice versa"""
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        # Add reverse mapping
        reverse_mapping = {v: k for k, v in mapping.items()}
        mapping.update(reverse_mapping)
        return mapping

    
    def connect(self) -> bool:
        """Establish connection to MT5"""
        if not mt5.initialize():
            raise ConnectionError("MetaTrader5 initialization failed")
        self.connected = True
        logger.info("Successfully connected to MT5")
        return True

    def get_data(self, symbol: str, timeframe: Union[str, int], num_candles: int) -> pd.DataFrame:
        # Convert integer timeframe to string if needed
        original_timeframe = timeframe
        if isinstance(timeframe, int):
            if timeframe not in self.timeframe_mapping:
                logger.error(f"Invalid timeframe provided: {timeframe}")
                raise DataFetchError(f"Invalid timeframe: {timeframe}")
            timeframe = self.timeframe_mapping[timeframe]

        # Check if timeframe is valid
        if timeframe not in self.timeframe_mapping:
            logger.error(f"Invalid timeframe provided: {timeframe}")
            raise DataFetchError(f"Invalid timeframe: {timeframe}")
        
        # Try cache first
        cached_data = self.cache.get_cached_data(symbol, timeframe, num_candles)
        if cached_data is not None:
            return cached_data

        # Ensure connection
        if not self.connected:
            self.connect()

        try:
            # Get MT5 timeframe constant
            tf_constant = self.timeframe_mapping[timeframe]

            # Fetch data from MT5
            rates = mt5.copy_rates_from_pos(symbol, tf_constant, 0, num_candles)
            if rates is None:
                raise DataFetchError(f"Failed to fetch data for {symbol}: {mt5.last_error()}")
            
            # Convert to DataFrame and validate
            df = pd.DataFrame(rates)
            validated_data = self.validator.validate_data(df)

            # Check if validated_data is None or empty before updating the cache
            if validated_data is None or validated_data.empty:
                logger.error(f"Validation failed for {symbol} {timeframe}, no valid data available.")
                raise DataFetchError(f"Validation failed for {symbol} {timeframe}, no valid data available.")
            
            # Store string representation of timeframe for cache key
            cache_timeframe = (
                timeframe if isinstance(original_timeframe, str) 
                else self.timeframe_mapping[original_timeframe]
            )
            
            # Update cache with the validated data
            self.cache.update_cache(
                symbol=symbol,
                timeframe=cache_timeframe,
                num_candles=num_candles,
                data=validated_data  # Ensure validated_data is passed here
            )
            
            logger.info(f"Successfully fetched and validated data for {symbol} {timeframe}")
            return validated_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
            raise DataFetchError(f"Error fetching data: {str(e)}")
        
    def disconnect(self):
        """Disconnect from MT5"""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("Successfully disconnected from MT5")
        except Exception as e:
            logger.error(f"Error during disconnection: {str(e)}")

    def get_symbols(self) -> List[str]:
        """Get list of available symbols"""
        if not self.connected:
            self.connect()
        
        symbols = mt5.symbols_get()
        return [symbol.name for symbol in symbols]

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed information about a symbol"""
        if not self.connected:
            self.connect()
            
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol {symbol} not found")
            
        return {
            'name': info.name,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_mode': info.trade_mode,
            'tick_size': info.tick_size,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max
        }