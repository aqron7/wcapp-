"""Weather adjustment for run/goal totals (free Open-Meteo, no key).

Warm air carries the ball, so MLB run scoring rises with temperature; heavy rain
suppresses scoring (or postpones). We apply a modest, transparent multiplier to
the expected total. We deliberately do NOT use wind here: its effect depends on
direction relative to park orientation, which we can't resolve reliably, so a
naive wind term is as likely to be wrong as right.

Coefficients are small and unvalidated — like everything else, they must earn
their keep through the CLV gate. Guarded: any failure returns a neutral 1.0.
"""

from __future__ import annotations

import requests

# MLB ballpark coordinates by home-team Kalshi code (lat, lon).
PARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667), "ATL": (33.8907, -84.4677), "BAL": (39.2839, -76.6217),
    "BOS": (42.3467, -71.0972), "CHC": (41.9484, -87.6553), "CWS": (41.8300, -87.6339),
    "CIN": (39.0975, -84.5069), "CLE": (41.4962, -81.6852), "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485), "HOU": (29.7572, -95.3556), "KC": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400), "MIA": (25.7781, -80.2197),
    "MIL": (43.0280, -87.9712), "MIN": (44.9817, -93.2776), "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262), "OAK": (37.7516, -122.2005), "ATH": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665), "PIT": (40.4469, -80.0057), "SD": (32.7073, -117.1566),
    "SF": (37.7786, -122.3893), "SEA": (47.5914, -122.3325), "STL": (38.6226, -90.1928),
    "TB": (27.7682, -82.6534), "TEX": (32.7473, -97.0825), "TOR": (43.6414, -79.3894),
    "WSH": (38.8730, -77.0074),
}


def totals_multiplier(temp_f: float, precip_mm: float) -> float:
    """Multiplier on expected runs from temperature + precipitation."""
    mult = 1.0 + (temp_f - 70.0) * 0.004    # ~+0.4% per °F above 70
    if precip_mm and precip_mm > 0.5:
        mult -= 0.05                          # meaningful rain suppresses scoring
    return max(0.90, min(1.10, mult))


class WeatherProvider:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._cache: dict[tuple, float] = {}

    def multiplier(self, park_code: str, date: str, hour: int) -> float:
        """Totals multiplier for a park at a date/hour; 1.0 if unavailable."""
        coords = PARK_COORDS.get(park_code)
        if not coords:
            return 1.0
        key = (park_code, date, hour)
        if key in self._cache:
            return self._cache[key]
        try:
            lat, lon = coords
            r = self._session.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation",
                "temperature_unit": "fahrenheit",
                "start_date": date, "end_date": date}, timeout=15)
            r.raise_for_status()
            h = r.json()["hourly"]
            times, temps, precs = h["time"], h["temperature_2m"], h["precipitation"]
            idx = min(range(len(times)), key=lambda i: abs(int(times[i][11:13]) - hour))
            mult = totals_multiplier(temps[idx], precs[idx])
        except Exception:  # noqa: BLE001
            mult = 1.0
        self._cache[key] = mult
        return mult
