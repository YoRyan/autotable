import datetime as dt
import re
import yaml
from collections import Counter, namedtuple
from functools import lru_cache
from pathlib import Path

import gtfs_kit as gk
import pyproj as pp

from mstsinstall import MSTSInstall, Route


class Timetable:
    def __init__(self, route: Route, date: dt.date, name: str):
        self.route = route
        self.date = date
        self.name = name
        self.trips = {} # (gtfs path, trip id) -> Trip
        self.station_commands = {}

Trip = namedtuple('Trip', ['stops', 'path', 'consist',
                           'start_offset', 'dispose_commands'])

Stop = namedtuple('Stop', ['station', 'arrival', 'departure', 'commands'])


def load_config(fp, install: MSTSInstall) -> Timetable:
    yd = yaml.safe_load(fp)
    route = next(r for r in install.routes if r.id.lower() == yd['route'].lower())
    tt = Timetable(route, yd['date'], yd['name'])
    for block in yd['gtfs']:
        feed_path = block['filename']
        feed = _load_gtfs(feed_path)

        # Select all filtered trips.
        trips_sat = {}
        for idx, group in enumerate(block['groups']):
            for _, row in _select_trips(
                    feed, yd['date'], group['selection']).iterrows():
                trip_id = row['trip_id']
                if trip_id not in trips_sat:
                    trips_sat[(idx, trip_id)] = _stops_and_times(
                        feed, yd['date'], trip_id)

        # Collect and map all station names.
        feed_stops = feed.get_stops()
        all_stops = set()
        for _, stops in trips_sat.items():
            all_stops.update(set(stop_id for stop_id, arrival, departure in stops))
        station_map = _map_stations(route, feed, list(all_stops))

        # Add all Trips to Timetable.
        for (idx, trip_id), sat in trips_sat.items():
            stops = [Stop(station=station_map[stop_id],
                          arrival=arrival,
                          departure=departure,
                          commands='')
                     for stop_id, arrival, departure in sat
                     if station_map.get(stop_id, None) is not None]
            if len(stops) < 2:
                continue
            group = block['groups'][idx]
            tt.trips[(feed_path, trip_id)] = Trip(
                path=group['path'],
                consist=group['consist'],
                stops=stops,
                start_offset=group['start'],
                dispose_commands=group['dispose'])
    return tt


@lru_cache(maxsize=None)
def _load_gtfs(path: Path) -> gk.feed.Feed:
    return gk.read_gtfs(path, dist_units='m') # Units don't matter (for now?).


def _select_trips(feed: gk.feed.Feed, date: dt.date, filters: dict):
    sel = feed.get_trips(date=date.strftime('%Y%m%d'))
    for prop, regex in filters.items():
        if prop not in ['route_id', 'service_id', 'trip_id', 'trip_headsign',
                        'trip_short_name', 'direction_id', 'block_id',
                        'shape_id', 'wheelchair_accessible', 'bikes_allowed']:
            raise KeyError(f'not a valid GTFS trip attribute: {prop}')
        sel = sel[sel['trip_id'].str.match(regex)]
    return sel


def _stops_and_times(feed: gk.feed.Feed, date: dt.date, trip_id: str) -> list:
    stop_times = feed.get_stop_times(date=date.strftime('%Y%m%d'))
    trip = stop_times[stop_times['trip_id'] == trip_id]
    def parse_time(s):
        match = re.match(r'^(\d?\d):([012345]\d):([012345]\d)$', s)
        if not match:
            raise ValueError(f'invalid GTFS time: {s}')
        return dt.time(hour=int(match.group(1)) % 24,
                       minute=int(match.group(2)),
                       second=int(match.group(3)))
    return [(row['stop_id'],
             parse_time(row['arrival_time']), parse_time(row['departure_time']))
            for _, row in trip.sort_values(by='stop_sequence').iterrows()]


def _map_stations(route: Route, feed: gk.feed.Feed, stop_ids: list) -> dict:
    @lru_cache(maxsize=64)
    def tokens(s: str) -> list: return re.split('[ \t:;,-]+', s.lower())

    word_frequency = Counter(
        sum((tokens(s_name) for s_name in route.stations().keys()), []))
    def similarity(a: str, b: str) -> float:
        intersect = set(tokens(a)) & set(tokens(b))
        return sum(1/word_frequency[token] if token in word_frequency else 0.0
                   for token in intersect)

    geod = pp.Geod(ellps='WGS84')
    def dist_km(a: tuple, b: tuple) -> float:
        lat_a, lon_a = a
        lat_b, lon_b = b
        _azf, _azb, dist = geod.inv(lon_a, lat_a, lon_b, lat_b)
        return dist/1000.0

    feed_stops = feed.get_stops()
    station_map = {}
    for stop_id in stop_ids:
        _, stop = next(feed_stops[feed_stops['stop_id'] == stop_id].iterrows())
        latlon = (stop['stop_lat'], stop['stop_lon'])
        matches = [s_name for s_name, s_list in route.stations().items()
                   if (similarity(stop['stop_name'], s_name) >= 1.0
                       and dist_km(s_list[0].latlon, latlon) < 10)]
        n = len(matches)
        if n == 0:
            station_map[stop_id] = None
        elif n == 1:
            station_map[stop_id] = matches[0]
        else:
            raise KeyError(
                f"ambiguous station: '{name}' - candidates: {matches}")
    return station_map

