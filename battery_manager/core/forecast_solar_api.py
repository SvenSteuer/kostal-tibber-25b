#!/usr/bin/env python3
"""
Forecast.Solar Professional API Client

Fetches hourly solar production forecasts from forecast.solar API
Supports multiple planes (roof orientations) and caching
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ForecastSolarAPI:
    """Client for forecast.solar Professional API"""

    def __init__(self, api_key: str, latitude: float, longitude: float):
        """
        Initialize forecast.solar API client

        Args:
            api_key: forecast.solar Professional API key
            latitude: Location latitude
            longitude: Location longitude
        """
        self.api_key = api_key
        self.latitude = latitude
        self.longitude = longitude
        self.base_url = "https://api.forecast.solar"

        # Cache for API responses (15 min cache)
        self._cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(minutes=15)

        logger.info(f"Forecast.Solar API initialized (lat={latitude}, lon={longitude})")

    def _build_url(self, endpoint: str, declination: int, azimuth: int, kwp: float) -> str:
        """
        Build API URL for a single plane

        Args:
            endpoint: API endpoint (e.g., 'estimate')
            declination: Roof tilt angle (0-90°)
            azimuth: Roof orientation (-180 to 180°, 0=South, 90=West, -90=East)
            kwp: Peak power in kWp

        Returns:
            Complete API URL
        """
        # Convert float to URL-safe format
        lat = str(self.latitude).replace('.', ',')
        lon = str(self.longitude).replace('.', ',')
        kwp_str = str(kwp).replace('.', ',')

        url = (f"{self.base_url}/{self.api_key}/{endpoint}/"
               f"{lat}/{lon}/{declination}/{azimuth}/{kwp_str}")

        return url

    def get_hourly_forecast(self,
                           planes: list,
                           days: int = 1) -> Dict[int, float]:
        """
        Get hourly solar production forecast for today

        Args:
            planes: List of dicts with 'declination', 'azimuth', 'kwp'
                   e.g., [{'declination': 22, 'azimuth': 45, 'kwp': 8.96}]
            days: Number of days to forecast (default: 1 = today only)

        Returns:
            dict: {hour: kwh_forecast} for each hour (0-23)
        """
        # Check cache first
        if self._is_cache_valid():
            logger.debug("Using cached forecast.solar data")
            return self._cache.get('hourly_forecast', {})

        try:
            today = datetime.now().astimezone().date()
            hourly_forecast = {}

            # Fetch forecast for each plane and combine
            for i, plane in enumerate(planes):
                logger.debug(f"Fetching forecast for plane {i+1}: "
                           f"azimuth={plane['azimuth']}°, "
                           f"tilt={plane['declination']}°, "
                           f"kWp={plane['kwp']}")

                url = self._build_url(
                    endpoint='estimate/watthours',
                    declination=plane['declination'],
                    azimuth=plane['azimuth'],
                    kwp=plane['kwp']
                )

                logger.info(f"Fetching Plane {i+1} from Forecast.Solar: {url}")

                response = requests.get(url, timeout=10)

                if response.status_code != 200:
                    logger.error(f"Forecast.Solar API error: HTTP {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    continue

                data = response.json()

                # Extract watt_hours data
                # The /estimate/watthours endpoint returns timestamps directly in 'result'
                if 'result' in data:
                    watt_hours = data['result']

                    # Filter out non-timestamp keys (API might include metadata)
                    valid_entries = {k: v for k, v in watt_hours.items()
                                   if isinstance(k, str) and len(k) >= 10}  # Timestamp format check

                    logger.info(f"Plane {i+1}: received {len(valid_entries)} time intervals")

                    # CRITICAL: Forecast.Solar /watthours endpoint returns CUMULATIVE values
                    # We need to convert each plane's cumulative values to hourly deltas first,
                    # then combine the deltas from all planes

                    # Step 1: Collect cumulative values for THIS plane, sorted by hour
                    plane_cumulative = {}
                    for timestamp_str, wh_value in valid_entries.items():
                        try:
                            # Parse timestamp (format: "2025-11-05 14:00:00")
                            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                            # Only process today's data (or up to 'days' ahead)
                            if dt.date() == today:
                                hour = dt.hour
                                kwh = float(wh_value) / 1000.0  # Wh to kWh
                                plane_cumulative[hour] = kwh

                        except (ValueError, TypeError) as e:
                            logger.debug(f"Skipping entry {timestamp_str}: {e}")
                            continue

                    # Step 2: Convert THIS plane's cumulative values to hourly deltas
                    if plane_cumulative:
                        sorted_hours = sorted(plane_cumulative.keys())
                        logger.debug(f"Plane {i+1}: Converting cumulative to hourly deltas for hours {sorted_hours}")

                        for idx, hour in enumerate(sorted_hours):
                            if idx == 0:
                                # First hour: use cumulative value as-is (production from midnight to this hour)
                                hourly_delta = plane_cumulative[hour]
                            else:
                                # Subsequent hours: subtract previous cumulative from current
                                prev_hour = sorted_hours[idx - 1]
                                hourly_delta = plane_cumulative[hour] - plane_cumulative[prev_hour]
                                hourly_delta = max(0.0, hourly_delta)  # Can't be negative

                            # Step 3: Add this plane's hourly delta to the combined forecast
                            hourly_forecast[hour] = hourly_forecast.get(hour, 0.0) + hourly_delta

                        logger.debug(f"Plane {i+1}: Converted {len(plane_cumulative)} cumulative values to hourly deltas")
                else:
                    logger.error(f"Plane {i+1}: No 'result' key in API response")
                    logger.error(f"Full response: {data}")

            if hourly_forecast:
                logger.info(f"✓ Forecast.Solar: Retrieved {len(hourly_forecast)} hours from API")
                logger.debug(f"Hourly forecast (kWh): {hourly_forecast}")

                # Update cache
                self._cache = {'hourly_forecast': hourly_forecast}
                self._cache_timestamp = datetime.now()
            else:
                logger.warning("No hourly forecast data retrieved from Forecast.Solar API")

            return hourly_forecast

        except requests.RequestException as e:
            logger.error(f"Network error calling Forecast.Solar API: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error getting hourly forecast from Forecast.Solar: {e}", exc_info=True)
            return {}

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if not self._cache or not self._cache_timestamp:
            return False

        age = datetime.now() - self._cache_timestamp
        return age < self._cache_duration

    def clear_cache(self):
        """Clear cached forecast data"""
        self._cache = {}
        self._cache_timestamp = None
        logger.debug("Forecast.Solar cache cleared")
