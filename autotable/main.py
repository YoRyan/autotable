# -*- coding: utf-8 -*-
import csv
import datetime as dt
import re
from argparse import ArgumentParser
from collections import Counter, defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, takewhile, tee
from pathlib import Path
from tempfile import NamedTemporaryFile

import gtfs_kit as gk
import more_itertools as mit
import pandas as pd
import pyproj as pp
import requests
import yaml
from gtfs_kit.helpers import weekday_to_str

import autotable.mstsinstall as msts
from autotable import __version__


_GTFS_UNITS = 'm'


@dataclass
class Trip:
    name: str
    stops: list
    path: msts.Route.Path
    consist: list
    start_offset: int
    start_commands: str
    note: str
    speed_mps: str
    speed_kph: str
    speed_mph: str
    delay_commands: str
    station_commands: dict
    dispose_commands: str

    def __post_init__(self):
        first_stop = self.stops[0].arrival
        # https://stackoverflow.com/a/656394
        start_datetime = dt.datetime.combine(
            dt.date(year=2000, month=1, day=1), first_stop)
        self.start_time = \
            (start_datetime + dt.timedelta(seconds=self.start_offset)).time()


@dataclass
class _SubConsist:
    consist: msts.Consist
    reverse: False

    def __str__(self):
        if re.match(r'[\+\$]', self.consist.id):
            if self.reverse:
                return f'<{self.consist.id}>$reverse'
            else:
                return f'<{self.consist.id}>'
        elif self.reverse:
            return f'{self.consist.id} $reverse'
        else:
            return self.consist.id


TripStop = namedtuple('TripStop', ['station', 'mapped_stop_id', 'mapped_stop_name',
                                   'arrival', 'departure'])


@dataclass
class _TripConfig:
    path: msts.Route.Path
    consist: list
    start_offset: int
    start_commands: str
    note: str
    speed_mps: str
    speed_kph: str
    speed_mph: str
    delay_commands: str
    dispose_commands: str
    station_commands: dict
    station_map: dict


class Timetable:

    def __init__(self, route: msts.Route, date: dt.date, name: str):
        self.route = route
        self.date = date
        self.name = name
        self.trips = []
        self.station_commands = {}

    def write_csv(self, fp):
        # csv settings per the May 2017 timetable document
        # http://www.elvastower.com/forums/index.php?/topic/30326-update-timetable-mode-signalling/
        writer = csv.writer(fp, delimiter='\t', quoting=csv.QUOTE_NONE)

        ordered_stations = Timetable._order_stations(
            self.trips, iter(self.route.stations().keys()))
        ordered_trips = self.trips

        def strftime(t: dt.time) -> str: return t.strftime('%H:%M')

        def start_col(trip: Trip) -> str:
            if trip.start_commands:
                return f'{strftime(trip.start_time)} {trip.start_commands}'
            else:
                return strftime(trip.start_time)

        def consist_col(trip: Trip) -> str:
            return '+'.join(str(subconsist) for subconsist in trip.consist)

        writer.writerow(chain(iter(('', '', '#comment')),
                              (trip.name for trip in ordered_trips)))
        writer.writerow(iter(('#comment', '', self.name)))
        writer.writerow(chain(iter(('#path', '', '')),
                              (trip.path.id for trip in ordered_trips)))
        writer.writerow(chain(iter(('#consist', '', '')),
                              (consist_col(trip) for trip in ordered_trips)))
        writer.writerow(chain(iter(('#start', '', '')),
                              (start_col(trip) for trip in ordered_trips)))
        writer.writerow(chain(iter(('#note', '', '')),
                              (trip.note for trip in ordered_trips)))
        writer.writerow(chain(iter(('#speed', '', '')),
                              (trip.speed_mps for trip in ordered_trips)))
        writer.writerow(chain(iter(('#speedkph', '', '')),
                              (trip.speed_kph for trip in ordered_trips)))
        writer.writerow(chain(iter(('#speedmph', '', '')),
                              (trip.speed_mph for trip in ordered_trips)))
        writer.writerow(chain(iter(('#restartdelay', '', '')),
                              (trip.delay_commands for trip in ordered_trips)))

        stops_index = {}
        for i, trip in enumerate(ordered_trips):
            for stop in trip.stops:
                stops_index[(i, stop.station)] = stop

        def station_stops(s_name: str):
            for i, trip in enumerate(ordered_trips):
                stop = stops_index.get((i, s_name), None)
                if stop is None:
                    yield ''
                    continue

                if (stop.arrival.hour == stop.departure.hour
                        and stop.arrival.minute == stop.departure.minute):
                    time = strftime(stop.arrival)
                else:
                    time = f'{strftime(stop.arrival)}-{strftime(stop.departure)}'

                commands = trip.station_commands.get(s_name,
                    trip.station_commands.get('', ''))
                if commands:
                    yield f'{time} {commands}'
                else:
                    yield time

        def station_mappings(s_name: str):
            for i, trip in enumerate(ordered_trips):
                stop = stops_index.get((i, s_name), None)
                if stop is None:
                    yield ''
                else:
                    yield f'{stop.mapped_stop_id} - {stop.mapped_stop_name}'

        writer.writerow([])
        for s_name in ordered_stations:
            commands = self.station_commands.get(
                s_name, self.station_commands.get('', ''))
            writer.writerow(
                chain(iter((s_name, commands, '')), station_stops(s_name)))
            writer.writerow(
                chain(iter(('#comment', '', '',)), station_mappings(s_name)))
        writer.writerow([])

        writer.writerow(chain(iter(('#dispose', '', '')),
                              (trip.dispose_commands for trip in ordered_trips)))

    def _order_stations(trips: list, stations: iter) -> list:
        sm_trips = \
            Counter(tuple(stop.station for stop in trip.stops) for trip in trips)
        with ProcessPoolExecutor() as executor:
            order = (next(stations),)
            for station in stations:
                candidates = (order[0:i] + (station,) + order[i:]
                              for i in range(len(order) + 1))
                future_to_key = \
                    {executor.submit(Timetable._station_order_cost, cand, sm_trips):
                     cand for cand in candidates}
                best_future = max(
                    as_completed(future_to_key), key=lambda future: future.result())
                order = future_to_key[best_future]
            return order

    def _station_order_cost(compare_order: tuple, trip_orders: Counter) -> int:
        def trip_cost(compare_order: tuple, trip_order: tuple) -> int:
            compare_index = {station: i for i, station in enumerate(compare_order)}
            trip_index = {station: i for i, station in enumerate(trip_order)}

            trip_set = set(trip_order)
            common_stations = [s for s in compare_order if s in trip_set]
            if len(common_stations) == 0:
                length = 0
            else:
                length = mit.ilen(takewhile(lambda s: s != common_stations[-1],
                                            iter(compare_order)))

            forward = (
                (mit.quantify(trip_index[s1] < trip_index[s2]
                              for s1, s2 in mit.pairwise(common_stations))
                     - mit.quantify(trip_index[s1] > trip_index[s2]
                                    for s1, s2 in mit.pairwise(common_stations))),
                0,
                mit.quantify(compare_index[s1] + 1 == compare_index[s2]
                             for s1, s2 in mit.pairwise(common_stations)),
                (-mit.quantify(trip_index[s1] + 1 != trip_index[s2]
                               for s1, s2 in mit.pairwise(common_stations))
                     - length))
            backward = (
                (mit.quantify(trip_index[s1] > trip_index[s2]
                              for s1, s2 in mit.pairwise(common_stations))
                     - mit.quantify(trip_index[s1] < trip_index[s2]
                                    for s1, s2 in mit.pairwise(common_stations))),
                -1,
                mit.quantify(compare_index[s1] == compare_index[s2] + 1
                             for s1, s2 in mit.pairwise(common_stations)),
                (-mit.quantify(trip_index[s1] != trip_index[s2] + 1
                               for s1, s2 in mit.pairwise(common_stations))
                     - length))
            return max(forward, backward)

        trip_costs = (n*trip_cost(compare_order, trip_order)
                      for trip_order, n in trip_orders.items())
        try:
            cost = next(trip_costs)
        except StopIteration:
            return 0.0
        for t in trip_costs:
            cost = tuple(x + y for x, y in zip(cost, t))
        return cost


_StopTime = namedtuple(
    '_StopTime', ['station', 'mapped_stop_id', 'arrival_days_elapsed', 'arrival',
                  'departure_days_elapsed', 'departure'])


def main():
    parser = ArgumentParser(
        description='Build Open Rails timetables with real-world GTFS data.')
    parser.add_argument('msts', type=Path,
                        help='path to MSTS installation or mini-route')
    parser.add_argument('yaml', type=Path,
                        help='path to timetable recipe file')
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s {__version__}')
    args = parser.parse_args()

    if 'version' in args:
        print(args.version)
        return

    with open(args.yaml, 'rt') as fp:
        timetable = load_config(fp, msts.MSTSInstall(args.msts), args.yaml.stem)
    with open(args.yaml.parent/f'{args.yaml.stem}.timetable-or', 'wt',
              newline='', encoding=msts.ENCODING) as fp:
        timetable.write_csv(fp)


def load_config(fp, install: msts.MSTSInstall, name: str) -> Timetable:
    yd = yaml.safe_load(fp)

    route_id = yd.get('route', None)
    if not isinstance(route_id, str):
        raise RuntimeError('route ID wasn\'t a string')
    route = install.routes[yd['route'].casefold()]

    date = yd.get('date', None)
    if not isinstance(date, dt.date):
        raise RuntimeError('date wasn\'t readable by PyYAML')

    tt = Timetable(route, date, name)
    for block in yd['gtfs']:
        if block.get('file', ''):
            feed_path = block['file']
            feed = _read_gtfs(feed_path)
        elif block.get('url', ''):
            feed_path = block['url']
            feed = _download_gtfs(feed_path)
        else:
            raise RuntimeError("GTFS block missing a 'file' or 'url'")

        # Read attributes from trip blocks.
        trip_configs = defaultdict(lambda: _TripConfig(
            path='',
            consist=[],
            start_offset=-120,
            start_commands='',
            note='',
            speed_mps='',
            speed_kph='',
            speed_mph='',
            delay_commands='',
            dispose_commands='',
            station_commands={},
            station_map={}))
        trip_groups = {}
        for i, group in enumerate(block['groups']):
            rows = _filter_trips(feed.trips, group.get('selection', {}))
            for _, trip_id in rows['trip_id'].iteritems():
                config = trip_configs[trip_id]
                if 'path' in group:
                    config.path = _parse_path(route, group['path'])
                if 'consist' in group:
                    config.consist = _parse_consist(install, group['consist'])
                config.start_offset = group.get('start_time', config.start_offset)
                config.start_commands = group.get('start', config.start_commands)
                config.note = group.get('note', config.note)
                config.speed_mps = group.get('speed_mps', config.speed_mps)
                config.speed_kph = group.get('speed_kph', config.speed_kph)
                config.speed_mph = group.get('speed_mph', config.speed_mph)
                config.delay_commands = group.get('delay', config.delay_commands)
                config.dispose_commands = \
                    group.get('dispose', config.dispose_commands)
                config.station_commands.update(
                    _strkeys(group.get('station_commands', {})))
                config.station_map.update(
                    _strkeys(group.get('station_map', {})))
                trip_groups[trip_id] = i

        # Group trips by trip block.
        grouped_trips = defaultdict(lambda: set())
        for trip_id, config in trip_configs.items():
            if config.path and config.consist:
                grouped_trips[trip_groups[trip_id]].add(trip_id)

        # Map GTFS stops to station names.
        auto_map = _map_stations(route, feed)
        gtfs_map = _strkeys(block.get('station_map', {}))
        def map_station(trip_id: str, stop_id: str) -> str:
            trip_map = trip_configs[trip_id]
            return trip_map.station_map.get(
                stop_id, gtfs_map.get(stop_id, auto_map.get(stop_id, None)))

        # Add all Trips to Timetable.
        feed_trips_indexed = feed.get_trips().set_index('trip_id')
        def make_trip(trip_id: str) -> Trip:
            def stop_name(stop_id: str) -> str:
                stops_indexed = feed.stops.set_index('stop_id')
                return stops_indexed.at[stop_id, 'stop_name']

            st1, st2, st3 = tee(_stop_times(feed, trip_id, map_station), 3)
            if mit.ilen(mit.take(2, st1)) < 2:
                return None

            config = trip_configs[trip_id]
            first_st = mit.first(st2)
            start_dt = dt.datetime.combine(
                date + dt.timedelta(days=-first_st.arrival_days_elapsed),
                first_st.arrival)
            start_dt += dt.timedelta(seconds=config.start_offset)
            if not _is_trip_start(feed, trip_id, start_dt.date()):
                return None

            def make_stop(st: _StopTime) -> TripStop:
                return TripStop(station=st.station,
                                mapped_stop_id=st.mapped_stop_id,
                                mapped_stop_name=stop_name(st.mapped_stop_id),
                                arrival=st.arrival,
                                departure=st.departure)

            trip = feed_trips_indexed.loc[trip_id]
            return Trip(
                name=f"{trip['trip_short_name']} {trip['trip_headsign']}",
                stops=[make_stop(st) for st in st3],
                path=config.path,
                consist=config.consist,
                start_offset=config.start_offset,
                start_commands=config.start_commands,
                note=config.note,
                speed_mps=config.speed_mps,
                speed_kph=config.speed_kph,
                speed_mph=config.speed_mph,
                delay_commands=config.delay_commands,
                station_commands=config.station_commands,
                dispose_commands=config.dispose_commands)
        for i in range(len(block['groups'])):
            trips = list(filter(lambda trip: trip is not None,
                                (make_trip(trip_id) for trip_id in grouped_trips[i])))
            trips.sort(key=lambda trip: trip.start_time)
            tt.trips += trips

    tt.station_commands = yd.get('station_commands', {})
    return tt


@lru_cache(maxsize=8)
def _read_gtfs(path: Path) -> gk.feed.Feed:
    return gk.read_gtfs(path, dist_units=_GTFS_UNITS)


@lru_cache(maxsize=8)
def _download_gtfs(url: str) -> gk.feed.Feed:
    tf = NamedTemporaryFile(delete=False)
    with requests.get(url, stream=True) as req:
        for chunk in req.iter_content(chunk_size=128):
            tf.write(chunk)
    tf.close()
    gtfs = gk.read_gtfs(tf.name, dist_units=_GTFS_UNITS)
    Path(tf.name).unlink()
    return gtfs


def _parse_path(route: msts.Route, yd: str) -> msts.Route.Path:
    paths = route.paths()
    path_id = yd.casefold()
    if path_id not in paths:
        raise RuntimeError(f"unknown path '{path_id}'")
    return paths[path_id]


def _parse_consist(install: msts.MSTSInstall, yd) -> list:
    if not isinstance(yd, list):
        return _parse_consist(install, [yd])

    consists = install.consists()
    def parse(subconsist: str) -> msts.Consist:
        split = subconsist.rsplit(maxsplit=1)
        reverse = len(split) == 2 and split[1].casefold() == '$reverse'
        con_id = (split[0] if reverse else subconsist).casefold()
        if con_id not in consists:
            raise RuntimeError(f"unknown consist '{con_id}'")
        return _SubConsist(consist=consists[con_id], reverse=reverse)

    return [parse(item) for item in yd]


def _filter_trips(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for prop, regex in filters.items():
        if prop not in ['route_id', 'service_id', 'trip_id', 'trip_headsign',
                        'trip_short_name', 'direction_id', 'block_id',
                        'shape_id', 'wheelchair_accessible', 'bikes_allowed']:
            raise KeyError(f'not a valid GTFS trip attribute: {prop}')
        df = df[df[prop].astype(str).str.match(regex)]
    return df


def _is_trip_start(feed: gk.feed.Feed, trip_id: str, date: dt.date) -> bool:
    """Equivalent to gtfs_kit.trips.is_trip_active(), but with our own fixes."""

    def parse_date(s: str) -> dt.date:
        return dt.datetime.strptime(s.strip(), '%Y%m%d').date()

    calendar_dates = \
        (feed.calendar_dates.copy() if feed.calendar_dates is not None
         else pd.DataFrame(columns=['service_id', 'date', 'exception_type']))
    calendar_dates['_date_parsed'] = \
        calendar_dates['date'].astype(str).apply(parse_date)
    calendar_dates = calendar_dates.set_index(['service_id', '_date_parsed'])
    exceptions = calendar_dates['exception_type'].astype(int)

    calendar = \
        (feed.calendar.set_index('service_id') if feed.calendar is not None
         else pd.DataFrame(columns=['service_id', 'start_date', 'end_date']))
    # Sometimes, pandas interprets dates as integers, not strings.
    in_service = calendar[['start_date', 'end_date']].astype(str)
    weekdays = calendar.drop(['start_date', 'end_date'], axis=1).astype(bool)

    trip = feed.trips.loc[feed.trips['trip_id'].astype(str) == trip_id].iloc[0]
    service_id = trip['service_id']
    if (service_id, date) in calendar_dates.index:
        exception = exceptions.at[(service_id, date), 'exception_type']
        if exception == 1:
            return True
        elif exception == 2:
            return False
        else:
            assert False
    elif service_id in calendar.index:
        in_range = parse_date(in_service.at[service_id, 'start_date']) \
            <= date \
            <= parse_date(in_service.at[service_id, 'end_date'])
        day_match = weekdays.at[service_id, weekday_to_str(date.weekday())]
        return in_range and day_match
    else:
        return False


def _stop_times(feed: gk.feed.Feed, trip_id: str, map_station) -> iter:
    trip = (feed.stop_times[feed.stop_times['trip_id'] == trip_id]
        .sort_values(by='stop_sequence'))
    times = trip[['arrival_time', 'departure_time']].astype(str)
    stop_ids = trip['stop_id'].astype(str)

    def parse_stop(idx: int) -> _StopTime:
        stop_id = stop_ids.iat[idx]
        station = map_station(trip_id, stop_id)
        if not station:
            return None

        arrival_days, arrival = parse_time(times.iloc[idx]['arrival_time'])
        departure_days, departure = parse_time(times.iloc[idx]['departure_time'])
        return _StopTime(station=station,
                         mapped_stop_id=stop_id,
                         arrival_days_elapsed=arrival_days,
                         arrival=arrival,
                         departure_days_elapsed=departure_days,
                         departure=departure)

    def parse_time(s):
        match = re.match(r'(\d?\d)\s*:\s*([012345]?\d)\s*:\s*([012345]?\d)', s)
        if not match:
            raise ValueError(f'invalid GTFS time: {s}')
        hours = int(match.group(1))
        return (hours//24, dt.time(hour=hours%24,
                                   minute=int(match.group(2)),
                                   second=int(match.group(3))))

    return filter(lambda st: st is not None,
                  (parse_stop(i) for i in range(len(trip))))


def _map_stations(route: msts.Route, feed: gk.feed.Feed) -> dict:
    @lru_cache(maxsize=64)
    def tokens(s: str) -> list: return re.split('[ \t:;,-]+', s.casefold())

    word_frequency = Counter(
        chain(*(tokens(s_name) for s_name in route.stations().keys())))
    def similarity(a: str, b: str) -> float:
        ta = set(tokens(a))
        tb = set(tokens(b))
        intersect = set(ta & tb)
        difference = set.symmetric_difference(ta, tb)
        return -0.1*len(difference) \
            + sum(1/word_frequency[token] if token in word_frequency else 0.0
                  for token in intersect)

    geod = pp.Geod(ellps='WGS84')
    def dist_km(a: tuple, b: tuple) -> float:
        lat_a, lon_a = a
        lat_b, lon_b = b
        _, _, dist = geod.inv(lon_a, lat_a, lon_b, lat_b)
        return dist/1000.0

    def map_station(stop: pd.Series) -> str:
        latlon = (stop['stop_lat'], stop['stop_lon'])
        matches = \
            {s_name: similarity(stop['stop_name'], s_name)
             for s_name, s_list in route.stations().items()
             if any(dist_km(platform.latlon, latlon) < 10.0 for platform in s_list)}
        matches = {s_name: similarity for s_name, similarity in matches.items()
                   if similarity >= 0}
        if len(matches) == 0:
            return None
        else:
            return max(matches, key=matches.get)

    stops = feed.get_stops()
    stops = stops.set_index(stops['stop_id'].astype(str))
    stops['_mapped_station'] = stops.apply(map_station, axis=1)
    return stops.to_dict()['_mapped_station']


def _strkeys(d: dict) -> dict: return {str(k): v for k, v in d.items()}


if __name__ == '__main__':
    main()

