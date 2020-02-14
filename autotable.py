import datetime as dt
import re
import yaml
from collections import namedtuple
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

import gtfs_kit as gk

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


def load_config(fp, install):
    yd = yaml.safe_load(fp)
    route = next(r for r in install.routes if r.id.lower() == yd['route'].lower())
    tt = Timetable(route, yd['date'], yd['name'])
    for block in yd['gtfs']:
        feed_path = block['filename']
        feed = _load_gtfs(feed_path)
        feed_stops = feed.get_stops()
        for group in block['groups']:
            for _, row in _select_trips(
                    feed, yd['date'], group['selection']).iterrows():
                stops = []
                for stop_id, arrival, departure in _stops_and_times(
                        feed, yd['date'], row['trip_id']):
                    stop_name = feed_stops[feed_stops['stop_id'] == stop_id]\
                        ['stop_name'].values[0]
                    station = _map_station(stop_name, tt.route)
                    if station is None:
                        continue
                    stops.append(Stop(station=station, arrival=arrival,
                                      departure=departure, commands=''))
                if len(stops) < 2:
                    continue
                tr = Trip(
                    path=group['path'],
                    consist=group['consist'],
                    stops=stops,
                    start_offset=group['start'],
                    dispose_commands=group['dispose'])
                tt.trips[(feed_path, row['trip_id'])] = tr
    return tt


@lru_cache(maxsize=None)
def _load_gtfs(path: Path):
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


def _stops_and_times(feed: gk.feed.Feed, date: dt.date, trip_id: str):
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


@lru_cache(maxsize=None)
def _map_station(station: str, route: Route):
    matches = get_close_matches(station, route.stations())
    n = len(matches)
    if n == 0:
        return None
    elif n == 1:
        return matches[0]
    else:
        raise KeyError(
            f'ambiguous station: {station}\npotential candidates: {matches}')

