#!/usr/bin/env python3
"""
Tibber-basierte Lade-Optimierung
Portiert von Home Assistant Automationen
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class TibberOptimizer:
    """Smart charging optimization based on Tibber prices"""

    def __init__(self, config: Dict):
        self.threshold_1h = config.get('tibber_price_threshold_1h', 8) / 100
        self.threshold_3h = config.get('tibber_price_threshold_3h', 8) / 100
        self.charge_duration_per_10 = config.get('charge_duration_per_10_percent', 18)
        self.consumption_learner = None  # v0.4.0
        self.forecast_solar_api = None  # v0.9.2

    def set_consumption_learner(self, learner):
        """Set consumption learner for advanced optimization (v0.4.0)"""
        self.consumption_learner = learner
        logger.info("Consumption learner integrated into optimizer")

    def set_forecast_solar_api(self, api):
        """Set Forecast.Solar API client for PV forecasts (v0.9.2)"""
        self.forecast_solar_api = api
        logger.info("Forecast.Solar API integrated into optimizer")

    def get_hourly_pv_forecast(self, ha_client, config, include_tomorrow=False) -> Dict[int, float]:
        """
        Get hourly PV forecast (v0.9.2: now supports Forecast.Solar API)

        Priority:
        1. Forecast.Solar Professional API (if enabled and configured)
        2. Home Assistant sensors with wh_hours attribute (fallback)

        Args:
            ha_client: Home Assistant client instance
            config: Configuration dict with sensor names
            include_tomorrow: If True, returns 48h forecast (today 0-23 + tomorrow 24-47)

        Returns:
            dict: {hour: kwh_forecast} for each hour
                  If include_tomorrow=False: hour 0-23 (today only)
                  If include_tomorrow=True: hour 0-47 (today=0-23, tomorrow=24-47)
        """
        # v0.9.2 - Try Forecast.Solar API first if enabled
        if (self.forecast_solar_api and
            config.get('enable_forecast_solar_api', False)):

            logger.debug("Using Forecast.Solar Professional API for PV forecast")

            planes = config.get('forecast_solar_planes', [])
            if planes:
                try:
                    hourly_forecast = self.forecast_solar_api.get_hourly_forecast(planes)
                    if hourly_forecast:
                        return hourly_forecast
                    else:
                        logger.warning("Forecast.Solar API returned no data, falling back to sensors")
                except Exception as e:
                    logger.error(f"Error using Forecast.Solar API: {e}, falling back to sensors")
            else:
                logger.warning("Forecast.Solar API enabled but no planes configured")

        # Fallback: Use Home Assistant sensors (original v0.8.1 method)
        logger.debug(f"Using Home Assistant sensors for PV forecast (include_tomorrow={include_tomorrow})")
        hourly_forecast = {}

        # Get sensor names from config
        roof1_today_sensor = config.get('pv_production_today_roof1')
        roof2_today_sensor = config.get('pv_production_today_roof2')
        roof1_tomorrow_sensor = config.get('pv_production_tomorrow_roof1') if include_tomorrow else None
        roof2_tomorrow_sensor = config.get('pv_production_tomorrow_roof2') if include_tomorrow else None

        logger.debug(f"PV forecast sensors: roof1_today='{roof1_today_sensor}', roof2_today='{roof2_today_sensor}'")
        if include_tomorrow:
            logger.debug(f"Tomorrow sensors: roof1_tomorrow='{roof1_tomorrow_sensor}', roof2_tomorrow='{roof2_tomorrow_sensor}'")

        if not roof1_today_sensor and not roof2_today_sensor:
            logger.warning("No PV forecast sensors configured")
            return {}

        try:
            # Get today's and tomorrow's date for filtering
            now = datetime.now().astimezone()
            today = now.date()
            tomorrow = today + timedelta(days=1)

            # Process TODAY's sensors
            for roof_sensor in [roof1_today_sensor, roof2_today_sensor]:
                if not roof_sensor:
                    continue

                logger.debug(f"Fetching today attributes from {roof_sensor}")
                attrs = ha_client.get_attributes(roof_sensor)
                if attrs:
                    logger.debug(f"Today {roof_sensor} attributes keys: {list(attrs.keys())}")
                    if 'wh_hours' in attrs:
                        wh_hours = attrs['wh_hours']
                        logger.debug(f"Today {roof_sensor} wh_hours has {len(wh_hours)} entries")

                        for timestamp_str, wh_value in wh_hours.items():
                            try:
                                # Parse timestamp
                                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                                # Only process today's data
                                if dt.date() != today:
                                    continue

                                hour = dt.hour  # 0-23 for today
                                kwh = float(wh_value) / 1000.0  # Wh to kWh

                                # Add to hourly forecast
                                hourly_forecast[hour] = hourly_forecast.get(hour, 0.0) + kwh

                            except (ValueError, TypeError) as e:
                                logger.warning(f"Error parsing wh_hours entry {timestamp_str}: {e}")
                                continue
                    else:
                        logger.warning(f"Sensor {roof_sensor} has no 'wh_hours' attribute")
                else:
                    logger.warning(f"Could not get attributes for sensor {roof_sensor}")

            # Process TOMORROW's sensors (if requested)
            if include_tomorrow:
                for roof_sensor in [roof1_tomorrow_sensor, roof2_tomorrow_sensor]:
                    if not roof_sensor:
                        continue

                    logger.debug(f"Fetching tomorrow attributes from {roof_sensor}")
                    attrs = ha_client.get_attributes(roof_sensor)
                    if attrs:
                        logger.debug(f"Tomorrow {roof_sensor} attributes keys: {list(attrs.keys())}")
                        if 'wh_hours' in attrs:
                            wh_hours = attrs['wh_hours']
                            logger.debug(f"Tomorrow {roof_sensor} wh_hours has {len(wh_hours)} entries")

                            for timestamp_str, wh_value in wh_hours.items():
                                try:
                                    # Parse timestamp
                                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                                    # Only process tomorrow's data
                                    if dt.date() != tomorrow:
                                        continue

                                    hour = dt.hour + 24  # Offset by 24 hours: 24-47 for tomorrow
                                    kwh = float(wh_value) / 1000.0  # Wh to kWh

                                    # Add to hourly forecast
                                    hourly_forecast[hour] = hourly_forecast.get(hour, 0.0) + kwh

                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Error parsing wh_hours entry {timestamp_str}: {e}")
                                    continue
                        else:
                            logger.warning(f"Tomorrow sensor {roof_sensor} has no 'wh_hours' attribute")
                    else:
                        logger.warning(f"Could not get attributes for tomorrow sensor {roof_sensor}")

            if hourly_forecast:
                logger.info(f"Retrieved hourly PV forecast for {len(hourly_forecast)} hours")
                logger.debug(f"PV forecast: {hourly_forecast}")
            else:
                logger.warning("No hourly PV forecast data available")

            return hourly_forecast

        except Exception as e:
            logger.error(f"Error getting hourly PV forecast: {e}")
            return {}

    def find_optimal_charge_end_time(self, prices: List[Dict]) -> Optional[datetime]:
        """
        Findet den optimalen Zeitpunkt zum Beenden der Ladung.
        Das ist der Moment, an dem der Preis nach einer günstigen Phase wieder steigt.

        Args:
            prices: Liste von Preis-Dicts mit 'total', 'startsAt', 'level'

        Returns:
            datetime des optimalen Ladeendes oder None
        """
        # v0.3.3 - Use timezone-aware datetime for comparison
        now = datetime.now().astimezone()

        # Brauchen mindestens 6 Datenpunkte (3 zurück, aktuell, 2 voraus)
        if len(prices) < 6:
            logger.warning("Not enough price data for optimization")
            return None

        # Durchlaufe Preise ab Index 3 (brauchen 2h Historie)
        for i in range(3, len(prices) - 2):
            try:
                # Parse startsAt Zeit
                starts_at_str = prices[i]['startsAt']
                starts_at = datetime.fromisoformat(starts_at_str.replace('Z', '+00:00'))

                # Überspringe vergangene Zeiten
                if starts_at <= now:
                    continue

                # Hole Preise
                current_price = float(prices[i]['total'])
                price_1h_ago = float(prices[i-1]['total'])
                price_2h_ago = float(prices[i-2]['total'])
                price_1h_future = float(prices[i+1]['total'])
                price_2h_future = float(prices[i+2]['total'])

                # Berechne 3h Summen
                sum_3h_past = current_price + price_1h_ago + price_2h_ago
                sum_3h_future = current_price + price_1h_future + price_2h_future

                # Bedingung 1: Preis steigt um mehr als Schwelle zur vorherigen Stunde
                condition_1 = current_price > price_1h_ago * (1 + self.threshold_1h)

                # Bedingung 2: Nächste 3h Block teurer als vergangener 3h Block
                condition_2 = sum_3h_past < sum_3h_future * (1 + self.threshold_3h)

                if condition_1 and condition_2:
                    logger.info(f"Found optimal charge end time: {starts_at}")
                    logger.info(f"  Current price: {current_price:.4f}, 1h ago: {price_1h_ago:.4f}")
                    logger.info(f"  3h past sum: {sum_3h_past:.4f}, 3h future sum: {sum_3h_future:.4f}")
                    return starts_at

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing price data at index {i}: {e}")
                continue

        logger.info("No optimal charge end time found (prices stay low)")
        return None

    def calculate_charge_start_time(self,
                                     charge_end: datetime,
                                     current_soc: float,
                                     target_soc: float = 95) -> datetime:
        """
        Berechnet den Ladebeginn basierend auf SOC-Differenz.

        Args:
            charge_end: Gewünschter Ladezeitpunkt Ende
            current_soc: Aktueller SOC in %
            target_soc: Ziel-SOC in %

        Returns:
            datetime des Ladebeginns
        """
        # Berechne benötigte Ladung
        soc_diff = target_soc - current_soc

        if soc_diff <= 0:
            # Bereits voll genug
            return charge_end

        # Berechne Ladedauer in Minuten
        charge_duration_minutes = (soc_diff / 10) * self.charge_duration_per_10

        # Berechne Startzeit
        charge_start = charge_end - timedelta(minutes=charge_duration_minutes)

        logger.info(f"Calculated charge start: {charge_start}")
        logger.info(f"  SOC: {current_soc}% → {target_soc}% ({soc_diff}%)")
        logger.info(f"  Duration: {charge_duration_minutes:.0f} minutes")

        return charge_start

    def plan_daily_battery_schedule(self,
                                    ha_client,
                                    config,
                                    current_soc: float,
                                    prices: List[Dict]) -> Dict:
        """
        Plans 48-hour battery schedule using predictive optimization (v1.1.0 - extended to 2 days)

        Simulates 48 hours (today + tomorrow) hour-by-hour with consumption, PV, and prices.
        Identifies deficits and schedules charging at cheapest times BEFORE deficits.

        Args:
            ha_client: Home Assistant client for sensor data
            config: Configuration dict
            current_soc: Current battery SOC (%)
            prices: List of Tibber price data with datetime and total price (today + tomorrow)

        Returns:
            dict: {
                'hourly_soc': [float],  # Projected SOC for each hour (0-47: today=0-23, tomorrow=24-47)
                'hourly_charging': [float],  # Planned grid charging kWh per hour
                'hourly_pv': [float],  # PV production per hour
                'hourly_consumption': [float],  # Consumption per hour
                'charging_windows': [dict],  # Detailed charging plan
                'last_planned': str  # ISO timestamp of planning
            }
        """
        if not self.consumption_learner:
            logger.warning("No consumption learner available for daily planning")
            return None

        try:
            now = datetime.now().astimezone()
            today = now.date()
            current_hour = now.hour

            # Get battery parameters
            battery_capacity = config.get('battery_capacity', 10.6)  # kWh
            min_soc = config.get('auto_safety_soc', 20)  # %
            max_soc = config.get('auto_charge_below_soc', 95)  # %
            max_charge_power = config.get('max_charge_power', 3900) / 1000  # kW

            # 1. Collect hourly data for 48 hours (today + tomorrow)
            tomorrow = today + timedelta(days=1)
            hourly_consumption = []
            hourly_pv = []
            hourly_prices = []

            # Get PV forecast for 48 hours (today + tomorrow)
            pv_forecast = self.get_hourly_pv_forecast(ha_client, config, include_tomorrow=True)

            # Build hourly data arrays for 48 hours
            for hour in range(48):
                # Determine actual date and hour for this iteration
                if hour < 24:
                    # Today (hours 0-23)
                    actual_date = today
                    actual_hour = hour
                else:
                    # Tomorrow (hours 24-47)
                    actual_date = tomorrow
                    actual_hour = hour - 24

                # Consumption forecast (weekday-aware)
                consumption = self.consumption_learner.get_average_consumption(actual_hour, target_date=actual_date)
                hourly_consumption.append(consumption)

                # PV forecast (already has correct indexing: 0-23=today, 24-47=tomorrow)
                pv = pv_forecast.get(hour, 0.0)
                hourly_pv.append(pv)

                # Price (find matching hour from Tibber data)
                price = 0.30  # Default fallback
                for p in prices:
                    # Tibber uses 'startsAt' key, not 'datetime'
                    start_time = p.get('startsAt', '')
                    if start_time:
                        try:
                            price_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            price_dt = price_dt.astimezone()  # Convert to local timezone
                            if price_dt.hour == actual_hour and price_dt.date() == actual_date:
                                price = p.get('total', 0.30)
                                break
                        except Exception:
                            continue
                hourly_prices.append(price)

            logger.info(f"Planning 48h battery schedule for {today} and {tomorrow}")
            logger.debug(f"Hourly consumption (48h): {[f'{c:.2f}' for c in hourly_consumption]}")
            logger.debug(f"Hourly PV (48h): {[f'{p:.2f}' for p in hourly_pv]}")

            # 2. Simulate SOC without any grid charging (baseline) - 48 hours
            baseline_soc = [0.0] * 48
            soc_kwh = (current_soc / 100) * battery_capacity

            for hour in range(48):
                if hour < current_hour:
                    baseline_soc[hour] = current_soc  # Past hours: use current
                else:
                    # Future hours: simulate
                    net_energy = hourly_pv[hour] - hourly_consumption[hour]
                    soc_kwh += net_energy

                    # Clamp to battery limits
                    max_kwh = (max_soc / 100) * battery_capacity
                    min_kwh = (min_soc / 100) * battery_capacity
                    soc_kwh = max(min_kwh, min(max_kwh, soc_kwh))

                    baseline_soc[hour] = (soc_kwh / battery_capacity) * 100

            # 3. Identify deficit hours (where SOC falls below minimum) - 48 hours
            deficit_hours = []
            for hour in range(current_hour, 48):
                if baseline_soc[hour] <= min_soc + 5:  # 5% buffer
                    # Calculate how much energy is missing
                    current_kwh = (baseline_soc[hour] / 100) * battery_capacity
                    target_kwh = ((min_soc + 10) / 100) * battery_capacity  # Charge to min + 10%
                    deficit_kwh = target_kwh - current_kwh

                    deficit_hours.append({
                        'hour': hour,
                        'soc': baseline_soc[hour],
                        'deficit_kwh': max(0, deficit_kwh)
                    })

            logger.info(f"Found {len(deficit_hours)} deficit hours: {[d['hour'] for d in deficit_hours]}")

            # 4. Plan charging windows (cheapest hours BEFORE deficits) - 48 hours
            charging_windows = []
            hourly_charging = [0.0] * 48

            for deficit in deficit_hours:
                deficit_hour = deficit['hour']
                needed_kwh = deficit['deficit_kwh']

                if needed_kwh < 0.5:
                    continue  # Skip small deficits

                # Find available hours before deficit
                available_hours = []
                for h in range(current_hour, deficit_hour):
                    if hourly_charging[h] == 0:  # Not already planned
                        available_hours.append({
                            'hour': h,
                            'price': hourly_prices[h]
                        })

                # Sort by price (cheapest first)
                available_hours.sort(key=lambda x: x['price'])

                # Allocate charging to cheapest hours
                remaining_kwh = needed_kwh
                for slot in available_hours:
                    if remaining_kwh <= 0:
                        break

                    hour = slot['hour']
                    # Maximum charge per hour (1 hour at max power)
                    max_charge_per_hour = max_charge_power  # kWh (kW * 1h)
                    charge_kwh = min(remaining_kwh, max_charge_per_hour)

                    hourly_charging[hour] = charge_kwh
                    remaining_kwh -= charge_kwh

                    charging_windows.append({
                        'hour': hour,
                        'charge_kwh': charge_kwh,
                        'price': slot['price'],
                        'reason': f'Prepare for deficit at {deficit_hour}:00'
                    })

            logger.info(f"Planned {len(charging_windows)} deficit-based charging windows")

            # 4b. ECONOMIC OPTIMIZATION: Opportunistic charging at cheap prices (v1.0.9) - 48 hours
            # Only charge if economically beneficial AND battery won't be filled by PV anyway
            economic_threshold = 1.10  # Minimum 10% cost saving required
            negative_price_threshold = 0.0  # Charge if price <= 0 (we get paid!)

            for hour in range(current_hour, 48):
                # Skip if already charging in this hour
                if hourly_charging[hour] > 0:
                    continue

                # Simulate SOC at this hour with PV but WITHOUT new grid charging
                temp_soc_kwh = (current_soc / 100) * battery_capacity
                for h in range(current_hour, hour + 1):
                    if h < current_hour:
                        continue
                    # Include only: PV + existing planned charging - consumption
                    net = hourly_pv[h] + hourly_charging[h] - hourly_consumption[h]
                    temp_soc_kwh += net
                    temp_soc_kwh = max(0, min((max_soc / 100) * battery_capacity, temp_soc_kwh))

                soc_at_hour = (temp_soc_kwh / battery_capacity) * 100

                # Check available battery space (considering PV will fill it)
                available_space_kwh = ((max_soc - soc_at_hour) / 100) * battery_capacity

                # Skip if battery will be nearly full anyway
                if available_space_kwh < 0.5:
                    continue

                # Special case: NEGATIVE PRICES - always charge (we get paid!)
                if hourly_prices[hour] <= negative_price_threshold:
                    charge_kwh = min(available_space_kwh, max_charge_power)
                    hourly_charging[hour] = charge_kwh
                    charging_windows.append({
                        'hour': hour,
                        'charge_kwh': charge_kwh,
                        'price': hourly_prices[hour],
                        'reason': f'NEGATIVE PRICE: {hourly_prices[hour]*100:.2f} Ct/kWh (we get paid to charge!)'
                    })
                    logger.info(f"💰 Negative price at {hour}:00: Charging {charge_kwh:.2f} kWh @ {hourly_prices[hour]*100:.2f} Ct/kWh")
                    continue

                # Find future expensive hours where this stored energy would be used
                future_expensive_hours = []
                for future_h in range(hour + 1, 48):
                    if hourly_prices[future_h] > hourly_prices[hour] * economic_threshold:
                        future_expensive_hours.append(future_h)

                # Skip if no future expensive hours
                if not future_expensive_hours:
                    continue

                # Calculate average expensive price
                avg_expensive_price = sum(hourly_prices[h] for h in future_expensive_hours) / len(future_expensive_hours)

                # Economic decision: Is it worth charging now?
                cost_per_kwh = hourly_prices[hour]
                benefit_per_kwh = avg_expensive_price

                if benefit_per_kwh > cost_per_kwh * economic_threshold:
                    # Economically beneficial! But calculate optimal amount, not maximum

                    # Step 1: Find target hour (when to stop charging for)
                    # Either: PV becomes sufficient, cheaper hour arrives, or expensive hours begin
                    target_hour = 48

                    # Check if a cheaper hour is coming
                    for future_h in range(hour + 1, 48):
                        if hourly_prices[future_h] < cost_per_kwh * 0.98:  # 2% cheaper
                            target_hour = future_h
                            logger.debug(f"Found cheaper hour at {future_h}:00, stopping calculation there")
                            break

                    # Check when PV becomes sufficient (covers 80%+ of consumption)
                    for future_h in range(hour + 1, min(target_hour, 48)):
                        if hourly_pv[future_h] >= hourly_consumption[future_h] * 0.8:
                            target_hour = future_h
                            logger.debug(f"PV becomes sufficient at {future_h}:00, stopping calculation there")
                            break

                    # Limit to first expensive hour as maximum target
                    if future_expensive_hours:
                        target_hour = min(target_hour, max(future_expensive_hours))

                    # Step 2: Calculate energy deficit from charge hour to target hour
                    # Simulate forward from 'hour' to 'target_hour' WITHOUT this charge
                    deficit_kwh = 0
                    sim_soc_kwh = temp_soc_kwh  # Start with SOC at charge hour (already calculated above)

                    for h in range(hour, target_hour):
                        net = hourly_pv[h] + hourly_charging[h] - hourly_consumption[h]
                        sim_soc_kwh += net

                        # If SOC drops below minimum, we need to cover that deficit
                        min_kwh = (min_soc / 100) * battery_capacity
                        if sim_soc_kwh < min_kwh:
                            deficit_kwh += (min_kwh - sim_soc_kwh)
                            sim_soc_kwh = min_kwh

                        # Cap at max SOC
                        sim_soc_kwh = min((max_soc / 100) * battery_capacity, sim_soc_kwh)

                    # Step 3: Charge optimal amount (deficit + small buffer, minimum 1 kWh)
                    # If no deficit, still charge a reasonable amount for the economic opportunity
                    if deficit_kwh < 0.5:
                        # No immediate deficit, but economically beneficial
                        # Charge enough for the expensive hours (estimate: consumption during expensive hours)
                        estimated_usage = sum(hourly_consumption[h] - hourly_pv[h]
                                            for h in future_expensive_hours
                                            if hourly_consumption[h] > hourly_pv[h])
                        optimal_charge = max(1.0, min(estimated_usage * 0.5, max_charge_power))
                    else:
                        # Charge deficit + 15% buffer
                        optimal_charge = deficit_kwh * 1.15

                    charge_kwh = min(optimal_charge, available_space_kwh, max_charge_power)

                    hourly_charging[hour] = charge_kwh
                    charging_windows.append({
                        'hour': hour,
                        'charge_kwh': charge_kwh,
                        'price': cost_per_kwh,
                        'reason': f'Economic: Buy @ {cost_per_kwh*100:.2f} Ct/kWh, avoid @ {benefit_per_kwh*100:.2f} Ct/kWh later (until h{target_hour})'
                    })

                    logger.info(f"💡 Economic charging at {hour}:00: {charge_kwh:.2f} kWh @ {cost_per_kwh*100:.2f} Ct/kWh "
                              f"(saves {(benefit_per_kwh - cost_per_kwh)*100:.2f} Ct/kWh, bridges until {target_hour}:00)")

            logger.info(f"Total charging windows planned: {len(charging_windows)} "
                       f"({len([w for w in charging_windows if 'deficit' in w['reason'].lower()])} deficit, "
                       f"{len([w for w in charging_windows if 'Economic' in w['reason']])} economic)")

            # 5. Re-simulate SOC with planned charging - 48 hours
            final_soc = [0.0] * 48
            soc_kwh = (current_soc / 100) * battery_capacity

            for hour in range(48):
                if hour < current_hour:
                    final_soc[hour] = current_soc
                else:
                    # Add: PV production + grid charging
                    # Subtract: consumption
                    net_energy = hourly_pv[hour] + hourly_charging[hour] - hourly_consumption[hour]
                    soc_kwh += net_energy

                    # Clamp to battery limits
                    max_kwh = (max_soc / 100) * battery_capacity
                    min_kwh = (min_soc / 100) * battery_capacity
                    soc_kwh = max(min_kwh, min(max_kwh, soc_kwh))

                    final_soc[hour] = (soc_kwh / battery_capacity) * 100

            # 6. Return comprehensive 48-hour plan
            plan = {
                'hourly_soc': final_soc,
                'hourly_charging': hourly_charging,
                'hourly_pv': hourly_pv,
                'hourly_consumption': hourly_consumption,
                'hourly_prices': hourly_prices,
                'charging_windows': charging_windows,
                'last_planned': now.isoformat(),
                'total_charging_kwh': sum(hourly_charging),
                'min_soc_reached': min(final_soc[current_hour:]) if current_hour < 48 else current_soc
            }

            logger.info(f"48h plan complete: {len(charging_windows)} charge windows, "
                       f"total {plan['total_charging_kwh']:.2f} kWh, "
                       f"min SOC {plan['min_soc_reached']:.1f}%")

            return plan

        except Exception as e:
            logger.error(f"Error planning daily battery schedule: {e}", exc_info=True)
            return None

    def predict_short_term_deficit(self,
                                   ha_client,
                                   config,
                                   lookahead_hours: int = 3) -> Tuple[bool, float, str]:
        """
        Predicts short-term energy deficit using hourly PV forecast (v0.8.1)

        Uses granular hourly forecast from forecast.solar instead of broad daily check.
        More intelligent than the old 6:00-18:00 approach.

        Args:
            ha_client: Home Assistant client for fetching sensor data
            config: Configuration dict with sensor names
            lookahead_hours: How many hours to look ahead (default: 3)

        Returns:
            (has_deficit: bool, deficit_kwh: float, reason: str)
        """
        if not self.consumption_learner:
            logger.warning("No consumption learner available, using fallback")
            return False, 0.0, "No consumption learning data"

        try:
            now = datetime.now().astimezone()
            current_hour = now.hour

            # Get hourly PV forecast
            pv_forecast = self.get_hourly_pv_forecast(ha_client, config)

            if not pv_forecast:
                logger.warning("No PV forecast available, cannot predict deficit")
                return False, 0.0, "No PV forecast data"

            # Calculate consumption and PV production for next N hours
            total_consumption = 0.0
            total_pv = 0.0

            for i in range(lookahead_hours):
                future_hour = (current_hour + i) % 24
                future_date = (now + timedelta(hours=i)).date()

                # Get predicted consumption for this hour
                hour_consumption = self.consumption_learner.get_average_consumption(
                    future_hour,
                    target_date=future_date
                )
                total_consumption += hour_consumption

                # Get PV forecast for this hour
                hour_pv = pv_forecast.get(future_hour, 0.0)
                total_pv += hour_pv

                logger.debug(f"Hour {future_hour}: Consumption={hour_consumption:.2f} kWh, "
                           f"PV={hour_pv:.2f} kWh")

            # Calculate deficit
            deficit = total_consumption - total_pv
            has_deficit = deficit > 0.5  # At least 0.5 kWh gap

            reason = (f"Next {lookahead_hours}h: Consumption={total_consumption:.1f} kWh, "
                     f"PV={total_pv:.1f} kWh, Deficit={deficit:.1f} kWh")

            logger.info(f"Short-term deficit check: {reason}")

            return has_deficit, max(0, deficit), reason

        except Exception as e:
            logger.error(f"Error predicting short-term deficit: {e}")
            return False, 0.0, f"Error: {e}"

    def predict_energy_deficit(self,
                              pv_remaining: float,
                              current_hour: int = None) -> tuple[bool, float]:
        """
        Predicts if there will be an energy deficit based on consumption learning.

        DEPRECATED: Use predict_short_term_deficit() for more granular forecasting (v0.8.1)

        Args:
            pv_remaining: Expected PV production remaining today (kWh)
            current_hour: Current hour (0-23), defaults to now

        Returns:
            (has_deficit: bool, deficit_kwh: float)
        """
        if not self.consumption_learner:
            # Fallback to simple threshold
            return pv_remaining < 5, max(0, 5 - pv_remaining)

        try:
            if current_hour is None:
                current_hour = datetime.now().astimezone().hour

            # Predict consumption until evening (18:00)
            # This covers the critical morning period before PV ramps up
            target_hour = 18
            if current_hour >= target_hour:
                target_hour = 23  # Rest of day

            predicted_consumption = self.consumption_learner.predict_consumption_until(target_hour)

            # Simple deficit: consumption > PV remaining
            deficit = predicted_consumption - pv_remaining
            has_deficit = deficit > 0.5  # At least 0.5 kWh gap

            logger.debug(f"Energy balance: PV={pv_remaining:.1f} kWh, "
                        f"Consumption={predicted_consumption:.1f} kWh, "
                        f"Deficit={deficit:.1f} kWh")

            return has_deficit, max(0, deficit)

        except Exception as e:
            logger.error(f"Error predicting energy deficit: {e}")
            # Fallback
            return pv_remaining < 5, max(0, 5 - pv_remaining)

    def should_charge_now(self,
                         planned_start: Optional[datetime],
                         current_soc: float,
                         min_soc: float,
                         max_soc: float,
                         pv_remaining: float = None,
                         ha_client=None,
                         config: Dict = None) -> tuple[bool, str]:
        """
        Entscheidet ob jetzt geladen werden soll.

        Args:
            planned_start: Planned charging start time
            current_soc: Current battery SOC (%)
            min_soc: Minimum SOC threshold (%)
            max_soc: Maximum SOC threshold (%)
            pv_remaining: (DEPRECATED) Total PV remaining today - use ha_client + config instead
            ha_client: Home Assistant client for hourly forecast (v0.8.1)
            config: Configuration dict for sensor names (v0.8.1)

        Returns:
            (should_charge: bool, reason: str)
        """
        # v0.3.3 - Use timezone-aware datetime for comparison
        now = datetime.now().astimezone()

        # Sicherheit: SOC zu niedrig
        if current_soc < min_soc:
            return True, f"SOC below minimum ({current_soc}% < {min_soc}%)"

        # Bereits voll genug
        if current_soc >= max_soc:
            return False, f"Battery full ({current_soc}% >= {max_soc}%)"

        # v0.8.1 - Use short-term deficit prediction with hourly forecasts
        if ha_client and config:
            has_deficit, deficit_kwh, reason = self.predict_short_term_deficit(
                ha_client, config, lookahead_hours=3
            )

            if not has_deficit:
                # Sufficient energy expected for next 3 hours
                return False, f"No short-term deficit: {reason}"

            # Geplanter Ladezeitpunkt erreicht?
            if planned_start and now >= planned_start:
                return True, f"Planned charging time reached - {reason}"

            return False, f"Deficit detected but waiting for optimal window - {reason}"

        # Fallback to old method if no HA client provided
        elif pv_remaining is not None:
            # v0.4.0 - Check energy deficit based on consumption learning (DEPRECATED)
            has_deficit, deficit_kwh = self.predict_energy_deficit(pv_remaining)

            if not has_deficit:
                # Sufficient energy expected (PV covers consumption)
                return False, f"Energy balance positive (PV {pv_remaining:.1f} kWh covers consumption)"

            # Geplanter Ladezeitpunkt erreicht?
            if planned_start and now >= planned_start:
                return True, f"Planned charging time reached (deficit: {deficit_kwh:.1f} kWh)"

            return False, f"Waiting for optimal charging window (deficit: {deficit_kwh:.1f} kWh)"

        else:
            logger.warning("No PV data provided to should_charge_now, cannot make decision")
            return False, "Insufficient data for charging decision"
