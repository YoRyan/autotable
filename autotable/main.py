# -*- coding: utf-8 -*-
import csv
import datetime as dt
import re
from argparse import ArgumentParser
from collections import Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile

import gtfs_kit as gk
import more_itertools as mit
import pandas as pd
import pyproj as pp
import requests
import yaml

from autotable.mstsinstall import MSTSInstall, Route


_GTFS_UNITS = 'm'


@dataclass
class Trip:
    name: str
    stops: list
    path: str
    consist: str
    start_offset: int
    note: str
    dispose_commands: str

    def __post_init__(self):
        first_stop = self.stops[0].arrival
        # https://stackoverflow.com/a/656394
        start_datetime = dt.datetime.combine(
            dt.date(year=2000, month=1, day=1), first_stop)
        self.start_time = \
            (start_datetime + dt.timedelta(seconds=self.start_offset)).time()


class Timetable:

    def __init__(self, route: Route, date: dt.date, name: str):
        self.route = route
        self.date = date
        self.name = name
        self.trips = {} # (gtfs path, trip id) -> Trip
        self.station_commands = {}

    def write_csv(self, fp):
        # csv settings per the Open Rails manual, 11.2.1 "Data definition"
        # https://open-rails.readthedocs.io/en/latest/timetable.html#data-definition
        writer = csv.writer(fp, delimiter=';', quoting=csv.QUOTE_NONE)

        ordered_stations = Timetable._order_stations(
            self.trips.values(), iter(self.route.stations().keys()))
        ordered_trips = Timetable._order_trips(
            list(self.trips.values()), ordered_stations)

        def strftime(t: dt.time) -> str: return t.strftime('%H:%M')

        writer.writerow(chain(iter(('', '', '#comment')),
                              (trip.name for trip in ordered_trips)))
        writer.writerow(iter(('#comment', '', self.name)))
        writer.writerow(chain(iter(('#path', '', '')),
                              (trip.path for trip in ordered_trips)))
        writer.writerow(chain(iter(('#consist', '', '')),
                              (trip.consist for trip in ordered_trips)))
        writer.writerow(chain(iter(('#start', '', '')),
                              (strftime(trip.start_time) for trip in ordered_trips)))
        writer.writerow(chain(iter(('#note', '', '')),
                              (trip.note for trip in ordered_trips)))

        stops_index = {}
        for i, trip in enumerate(ordered_trips):
            for stop in trip.stops:
                stops_index[(i, stop.station)] = stop

        def station_stops(s_name: str):
            for i, trip in enumerate(ordered_trips):
                stop = stops_index.get((i, s_name), None)
                if stop is None:
                    yield ''
                elif (stop.arrival.hour == stop.departure.hour
                        and stop.arrival.minute == stop.departure.minute):
                    yield strftime(stop.arrival)
                else:
                    yield f'{strftime(stop.arrival)}-{strftime(stop.departure)}'

        writer.writerow([])
        for s_name in ordered_stations:
            writer.writerow(
                chain(iter((s_name, self.station_commands.get(s_name, ''), '')),
                      station_stops(s_name)))
        writer.writerow([])

        writer.writerow(chain(iter(('#dispose', '', '')),
                              (trip.dispose_commands for trip in ordered_trips)))

    def _order_stations(trips: list, stations: iter) -> list:
        sm_trips = \
            tuple(tuple(stop.station for stop in trip.stops) for trip in trips)
        with ProcessPoolExecutor() as executor:
            current_order = (next(stations),)
            candidates = set(stations)
            while len(candidates) > 0:
                prepend = (((cand,) + current_order, cand) for cand in candidates)
                append = ((current_order + (cand,), cand) for cand in candidates)

                future_to_key = {}
                for order, candidate in chain(prepend, append):
                    future = executor.submit(
                        Timetable._station_order_cost, order, sm_trips)
                    future_to_key[future] = (order, candidate)

                best_future = max(
                    as_completed(future_to_key), key=lambda future: future.result())
                current_order, selected = future_to_key[best_future]
                candidates.discard(selected)
            return current_order

    def _station_order_cost(compare_order: tuple, trip_orders: tuple) -> int:
        def trip_cost(compare_order: tuple, trip_order: tuple) -> int:
            compare_index = {station: i for i, station in enumerate(compare_order)}
            trip_index = {station: i for i, station in enumerate(trip_order)}

            trip_set = set(trip_order)
            common_stations = [s for s in compare_order if s in trip_set]

            discontinuous = mit.quantify(
                abs(compare_index[s1] - compare_index[s2]) > 1
                for s1, s2 in mit.pairwise(common_stations))
            forwards = mit.quantify(
                trip_index[s1] + 1 == trip_index[s2]
                for s1, s2 in mit.pairwise(common_stations))
            backwards = mit.quantify(
                trip_index[s1] == trip_index[s2] + 1
                for s1, s2 in mit.pairwise(common_stations))
            direction = int(forwards > backwards)
            return (max(forwards, backwards), direction, -discontinuous)

        cost = (0,)*3
        for t in (trip_cost(compare_order, trip_order)
                  for trip_order in trip_orders):
            cost = tuple(x + y for x, y in zip(cost, t))
        return cost

    def _order_trips(trips: list, station_order: list) -> list:
        order_index = dict((s_name, i) for i, s_name in enumerate(station_order))

        # TODO try to detect branching
        forwards = []
        backwards = []
        for trip in trips:
            def stations() -> iter: return (stop.station for stop in trip.stops)
            forward_pairs = mit.quantify(order_index[s1] < order_index[s2]
                                         for s1, s2 in mit.pairwise(stations()))
            backward_pairs = max(0, mit.ilen(stations()) - 1 - forward_pairs)
            if forward_pairs >= backward_pairs:
                forwards.append(trip)
            else:
                backwards.append(trip)

        forwards.sort(key=lambda trip: trip.start_time)
        backwards.sort(key=lambda trip: trip.start_time)
        return forwards + backwards


Stop = namedtuple('Stop', ['station', 'arrival', 'departure', 'commands'])


def main():
    parser = ArgumentParser(
        description='Build Open Rails timetables with real-world GTFS data.')
    parser.add_argument('msts', type=Path,
                        help='path to MSTS installation or mini-route')
    parser.add_argument('yaml', type=Path,
                        help='path to timetable recipe file')
    args = parser.parse_args()

    with open(args.yaml, 'rt') as fp:
        timetable = load_config(fp, MSTSInstall(args.msts), args.yaml.stem)
    with open(args.yaml.parent/f'{args.yaml.stem}.timetable_or', 'wt') as fp:
        timetable.write_csv(fp)


def load_config(fp, install: MSTSInstall, name: str) -> Timetable:
    yd = yaml.safe_load(fp)
    route = next(r for r in install.routes
                 if r.id.casefold() == yd['route'].casefold())
    route_paths = set(path.id.casefold() for path in route.paths())
    all_consists = set(consist.id.casefold() for consist in install.consists())
    tt = Timetable(route, yd['date'], name)
    for block in yd['gtfs']:
        if block.get('file', ''):
            feed_path = block['file']
            feed = _read_gtfs(feed_path)
        elif block.get('url', ''):
            feed_path = block['url']
            feed = _download_gtfs(feed_path)
        else:
            raise RuntimeError("GTFS block missing a 'file' or 'url'")

        # Validate the path and consist fields.
        for group in block['groups']:
            if ('path' in group
                    and group['path'].casefold() not in route_paths):
                raise RuntimeError(f"unknown {route.id} path '{group['path']}'")
            # TODO support the full syntax for timetable consists?
            elif ('consist' in group
                    and group['consist'].casefold() not in all_consists):
                raise RuntimeError(f"unknown consist '{group['consist']}'")

        # Select all filtered trips.
        feed_trips = _get_trips(feed, yd['date'])
        group_trips = \
            {i: set(trip_id for _, trip_id in _select(
                 feed_trips, group.get('selection', {}))['trip_id'].iteritems())
             for i, group in enumerate(block['groups'])}
        trips_sat = {trip_id: list(_stops_and_times(feed, trip_id))
                     for trip_id in chain(*group_trips.values())}

        # Collect and map all station names.
        all_stops = chain(*((stop_id for stop_id, _, _ in stops)
                            for stops in trips_sat.values()))
        station_map = _map_stations(route, feed, mit.unique_everseen(all_stops),
                                    init_map=block.get('station_map', {}))

        # Add all Trips to Timetable.
        for _, trip in feed_trips[feed_trips['trip_id']
                .isin(chain(*group_trips.values()))].iterrows():
            stops = [Stop(station=station_map[stop_id],
                          arrival=arrival,
                          departure=departure,
                          commands='')
                     for stop_id, arrival, departure in trips_sat[trip['trip_id']]
                     if station_map.get(stop_id, None) is not None]
            if len(stops) < 2:
                continue

            path = consist = note = dispose = ''
            start = -120
            for group in (group for i, group in enumerate(block['groups'])
                          if trip['trip_id'] in group_trips[i]):
                path = group.get('path', '')
                consist = group.get('consist', '')
                start = group.get('start', start)
                note = group.get('note', note)
                dispose = group.get('dispose', dispose)
            if not path:
                raise RuntimeError(f"trip {trip['trip_id']} is missing a path")
            elif not consist:
                raise RuntimeError(f"trip {trip['trip_id']} is missing a consist")
            tt.trips[(feed_path, trip['trip_id'])] = Trip(
                name=f"{trip['trip_short_name']} {trip['trip_headsign']}",
                path=path,
                consist=consist,
                stops=stops,
                start_offset=start,
                note=note,
                dispose_commands=dispose)
    return tt


@lru_cache(maxsize=8)
def _read_gtfs(path: Path) -> gk.feed.Feed:
    return gk.read_gtfs(path, dist_units=_GTFS_UNITS)


@lru_cache(maxsize=8)
def _download_gtfs(url: str) -> gk.feed.Feed:
    with NamedTemporaryFile() as tf:
        with requests.get(url, stream=True) as req:
            for chunk in req.iter_content(chunk_size=128):
                tf.write(chunk)
        tf.seek(0)
        return gk.read_gtfs(tf.name, dist_units=_GTFS_UNITS)


def _get_trips(feed: gk.feed.Feed, date: dt.date) -> pd.DataFrame:
    return feed.get_trips(date=date.strftime('%Y%m%d'))


def _select(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for prop, regex in filters.items():
        if prop not in ['route_id', 'service_id', 'trip_id', 'trip_headsign',
                        'trip_short_name', 'direction_id', 'block_id',
                        'shape_id', 'wheelchair_accessible', 'bikes_allowed']:
            raise KeyError(f'not a valid GTFS trip attribute: {prop}')
        df = df[df[prop].astype(str).str.match(regex)]
    return df


def _stops_and_times(feed: gk.feed.Feed, trip_id: str) -> iter:
    stop_times = feed.get_stop_times()
    trip = stop_times[stop_times['trip_id'] == trip_id]
    def parse_time(s):
        match = re.match(r'^(\d?\d):([012345]\d):([012345]\d)$', s)
        if not match:
            raise ValueError(f'invalid GTFS time: {s}')
        return dt.time(hour=int(match.group(1)) % 24,
                       minute=int(match.group(2)),
                       second=int(match.group(3)))
    return ((row['stop_id'],
             parse_time(row['arrival_time']), parse_time(row['departure_time']))
            for _, row in trip.sort_values(by='stop_sequence').iterrows())


def _map_stations(
        route: Route, feed: gk.feed.Feed, stop_ids: iter, init_map: dict={}) -> dict:
    @lru_cache(maxsize=64)
    def tokens(s: str) -> list: return re.split('[ \t:;,-]+', s.casefold())

    word_frequency = Counter(
        chain(*(tokens(s_name) for s_name in route.stations().keys())))
    def similarity(a: str, b: str) -> float:
        intersect = set(tokens(a)) & set(tokens(b))
        return sum(1/word_frequency[token] if token in word_frequency else 0.0
                   for token in intersect)

    geod = pp.Geod(ellps='WGS84')
    def dist_km(a: tuple, b: tuple) -> float:
        lat_a, lon_a = a
        lat_b, lon_b = b
        _, _, dist = geod.inv(lon_a, lat_a, lon_b, lat_b)
        return dist/1000.0

    def map_station(stop: pd.Series) -> str:
        if stop['stop_id'] in init_map:
            station = init_map[stop['stop_id']]
            if route.stations()[station]:
                return station
            else:
                raise RuntimeError(
                    f"specified station not present in {route.id}: '{station}'")
        else:
            latlon = (stop['stop_lat'], stop['stop_lon'])
            matches = {s_name: similarity(stop['stop_name'], s_name)
                       for s_name, s_list in route.stations().items()
                       if any(dist_km(platform.latlon, latlon) < 10.0
                              for platform in s_list)}
            if len(matches) == 0:
                return None
            else:
                return max(matches, key=matches.get)

    feed_stops = feed.get_stops()
    return {stop['stop_id']: map_station(stop) for _, stop
            in feed_stops[feed_stops['stop_id'].isin(stop_ids)].iterrows()}


if __name__ == '__main__':
    main()

