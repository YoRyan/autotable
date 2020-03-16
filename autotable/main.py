# -*- coding: utf-8 -*-
import datetime as dt
import re
import typing as typ
from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, tee
from multiprocessing import freeze_support
from pathlib import Path
from tempfile import NamedTemporaryFile

import gtfs_kit as gk # type: ignore
import pandas as pd # type: ignore
import pyproj as pp # type: ignore
import pytz
import requests
import yaml
from gtfs_kit.helpers import weekday_to_str # type: ignore
from more_itertools import first, ilen, take
from timezonefinder import TimezoneFinder # type: ignore

import autotable.mstsinstall as msts
import autotable.timetable as tt
from autotable import __version__


_GTFS_UNITS = 'm'
_MIN_STOPS = 1

_TzF = TimezoneFinder()


_StopId = str


_TripId = str


@dataclass
class _StopTime:
    station: msts.Station
    mapped_stop_id: _StopId
    arrival_days_elapsed: int
    arrival: dt.time
    departure_days_elapsed: int
    departure: dt.time


@dataclass
class _TripConfig:
    path: typ.Optional[msts.Route.TrainPath]
    consist: typ.Optional[typ.Sequence[tt.ConsistComponent]]
    start_offset: int
    start_commands: str
    note_commands: str
    speed_commands: str
    delay_commands: str
    station_commands: typ.Dict[msts.Station, str]
    dispose_commands: str
    station_map: typ.Dict[_StopId, msts.Station]

    def finalize(self, name: str, stops: typ.Sequence[tt.Stop]) -> tt.Trip:
        assert self.path is not None and self.consist is not None
        return tt.Trip(
            name=name,
            stops=stops,
            path=self.path,
            consist=self.consist,
            start_offset=self.start_offset,
            start_commands=self.start_commands,
            note_commands=self.note_commands,
            speed_commands=self.speed_commands,
            delay_commands=self.delay_commands,
            station_commands=self.station_commands,
            dispose_commands=self.dispose_commands)


def main() -> None:
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

    print('Loading MSTS/ORTS installation ...')
    install = msts.MSTSInstall(args.msts)

    print('Reading recipe ...')
    with open(args.yaml, 'rt') as fp:
        timetable = load_config(fp, install, args.yaml.stem)

    print(f'Route: {timetable.route.name}')
    print(f'Name: {timetable.name}')
    print(f'{len(timetable.trips)} total trips')
    print('Writing timetable ...')
    with open(args.yaml.parent/f'{args.yaml.stem}.timetable-or', 'wt',
              newline='', encoding=msts.ENCODING) as fp:
        timetable.write_csv(fp)


def load_config(fp: typ.TextIO, install: msts.MSTSInstall, name: str) \
        -> tt.Timetable:
    yd = yaml.safe_load(fp)

    # 'route'
    route_id = yd.get('route', None)
    if not isinstance(route_id, str):
        raise RuntimeError("'route' missing or not a string")
    try:
        route = install.route(route_id)
    except KeyError:
        raise RuntimeError(f"unknown route '{route_id}'")

    route_stations = set(route.station_names())
    def validate_station(station: str) -> None:
        if station not in route_stations:
            raise RuntimeError(f"{route.name} has no such station '{station}'")

    # 'date'
    date = yd.get('date', None)
    if not isinstance(date, dt.date):
        raise RuntimeError("'date' missing or not readable by PyYAML")

    # 'timezone'
    timezone = _parse_timezone(
        yd.get(
            'timezone',
            _TzF.certain_timezone_at(lat=route.latlon[0], lng=route.latlon[1])),
        date)

    # 'speed_unit'
    speed_unit_str = yd.get('speed_unit', None)
    if speed_unit_str == 'm/s' or speed_unit_str is None:
        speed_unit = tt.SpeedUnit.MS
    elif speed_unit_str == 'km/h' or speed_unit_str == 'kph':
        speed_unit = tt.SpeedUnit.KPH
    elif speed_unit_str == 'mi/h' or speed_unit_str == 'mph':
        speed_unit = tt.SpeedUnit.MPH
    else:
        raise ValueError(f"unknown unit of speed '{speed_unit_str}'")

    # 'station_commands'
    station_commands = yd.get('station_commands', {})
    for s_name, _ in station_commands.items():
        if s_name != '':
            validate_station(s_name)

    # 'gtfs'
    def load_gtfs(yd: typ.Mapping[str, typ.Any]) \
            -> typ.Generator[tt.Trip, None, None]:
        if 'file' in yd:
            feed_path = yd['file']
            feed = _read_gtfs(feed_path)
        elif 'url' in yd:
            feed_path = yd['url']
            feed = _download_gtfs(feed_path)
        else:
            raise RuntimeError("'gtfs' block missing a 'file' or 'url'")

        feed_stops_indexed = feed.get_stops().set_index('stop_id')
        def validate_stop(stop_id: _StopId) -> None:
            if stop_id not in feed_stops_indexed.index:
                raise RuntimeError(
                    f"unknown stop ID '{stop_id}' for the gtfs feed {feed_path}")

        def stop_name(stop_id: _StopId) -> str:
            return feed_stops_indexed.at[stop_id, 'stop_name']

        # Read attributes from trip blocks.
        prelim_trips: typ.Mapping[_StopId, _TripConfig] = \
            defaultdict(lambda: _TripConfig(
                path=None,
                consist=None,
                start_offset=-120,
                start_commands='',
                note_commands='',
                speed_commands='',
                delay_commands='',
                station_commands={},
                dispose_commands='',
                station_map={}))
        trip_groups: typ.Dict[_TripId, int] = {}
        for i, group in enumerate(yd.get('groups', [])):
            rows = _filter_trips(feed.trips, group.get('selection', {}))
            for _, trip_id in rows['trip_id'].iteritems():
                trip = prelim_trips[trip_id]

                if 'path' in group:
                    trip.path = _parse_path(route, group['path'])
                if 'consist' in group:
                    trip.consist = _parse_consist(install, group['consist'])

                trip.start_offset = group.get('start_time', trip.start_offset)
                trip.start_commands = group.get('start', trip.start_commands)
                trip.note_commands = group.get('note', trip.note_commands)
                trip.speed_commands = group.get('speed', trip.speed_commands)
                trip.delay_commands = group.get('delay', trip.delay_commands)
                trip.dispose_commands = group.get('dispose', trip.dispose_commands)

                station_commands = _strkeys(group.get('station_commands', {}))
                for s_name, _ in station_commands.items():
                    validate_station(s_name)
                trip.station_commands.update(station_commands)

                station_map = _strkeys(group.get('station_map', {}))
                for stop_id, _ in station_map.items():
                    validate_stop(stop_id)
                trip.station_map.update(station_map)

                trip_groups[_TripId(trip_id)] = i

        # Map GTFS stops to station names.
        auto_map = _map_stations(route, feed)
        gtfs_map = _strkeys(yd.get('station_map', {}))
        def map_station(trip_id: _TripId, stop_id: _StopId) -> msts.Station:
            return prelim_trips[trip_id].station_map.get(
                stop_id, gtfs_map.get(stop_id, auto_map.get(stop_id, None)))

        # Add all Trips to Timetable.
        feed_trips_indexed = feed.get_trips().set_index('trip_id')
        def finalize_trip(trip: _TripConfig, trip_id: _TripId) \
                -> typ.Optional[tt.Trip]:
            if not trip.path or not trip.consist:
                return None

            st1, st2, st3 = tee(_stop_times(feed, trip_id, map_station), 3)
            if ilen(take(_MIN_STOPS, st1)) < _MIN_STOPS:
                return None

            first_st = first(st2)
            start_dt = dt.datetime.combine(
                date + dt.timedelta(days=-first_st.arrival_days_elapsed),
                first_st.arrival)
            if not _is_trip_start(
                    feed, trip_id,
                    (start_dt + dt.timedelta(seconds=trip.start_offset)).date()):
                return None

            trip_row = feed_trips_indexed.loc[trip_id]
            route = feed.routes[feed.routes['route_id']
                                == trip_row['route_id']].squeeze()

            agency = feed.agency[feed.agency['agency_id']
                                 == route['agency_id']].squeeze()
            # Assume route and trip timezone are identical if the GTFS feed
            # doesn't specify one.
            trip_timezone = (_parse_timezone(agency['agency_timezone'], date)
                             if agency['agency_timezone'] else timezone)
            def make_stop(st: _StopTime) -> tt.Stop:
                start_date = start_dt.date()
                arrival_dt = dt.datetime.combine(
                    start_date + dt.timedelta(days=st.arrival_days_elapsed),
                    st.arrival,
                    tzinfo=trip_timezone)
                departure_dt = dt.datetime.combine(
                    start_date + dt.timedelta(days=st.departure_days_elapsed),
                    st.departure,
                    tzinfo=trip_timezone)
                return tt.Stop(
                    station=st.station,
                    comment=f'{st.mapped_stop_id} {stop_name(st.mapped_stop_id)}',
                    arrival=arrival_dt,
                    departure=departure_dt)

            return trip.finalize(_name_trip(feed, trip_id),
                                 [make_stop(st) for st in st3])

        grouped_trips = _reverse(trip_groups)
        for i in sorted(grouped_trips.keys()):
            trips = [trip for trip in (finalize_trip(prelim_trips[trip_id], trip_id)
                                       for trip_id in grouped_trips[i])
                     if trip is not None]
            trips.sort(key=tt.Trip.start_time)
            yield from trips

    gtfs = yd.get('gtfs', None)
    if not isinstance(gtfs, list):
        raise RuntimeError("'gtfs' not present or not a list of dictionaries")
    trips: typ.List[tt.Trip] = []
    for block in gtfs:
        if not isinstance(block, dict):
            raise RuntimeError("'gtfs' block wasn't a dictionary")
        trips.extend(load_gtfs(block))

    return tt.Timetable(name=name,
                        route=route,
                        date=date,
                        tzinfo=timezone,
                        trips=trips,
                        station_commands=station_commands,
                        speed_unit=speed_unit)


def _parse_timezone(tz: str, date: dt.date) -> dt.tzinfo:
    # Resolve daylight savings and other ambiguities.
    localized = \
        pytz.timezone(tz).localize(dt.datetime.combine(date, dt.time(6, 1)))
    if localized.tzinfo is None:
        raise RuntimeError(f"bad timezone: '{tz}'")
    return localized.tzinfo


@lru_cache(maxsize=8)
def _read_gtfs(path: Path) -> gk.feed.Feed:
    return gk.read_feed(path, dist_units=_GTFS_UNITS)


@lru_cache(maxsize=8)
def _download_gtfs(url: str) -> gk.feed.Feed:
    tf = NamedTemporaryFile(delete=False)
    with requests.get(url, stream=True) as req:
        for chunk in req.iter_content(chunk_size=128):
            tf.write(chunk)
    tf.close()
    gtfs = gk.read_feed(tf.name, dist_units=_GTFS_UNITS)
    Path(tf.name).unlink()
    return gtfs


def _parse_path(route: msts.Route, path_id: str) -> msts.Route.TrainPath:
    try:
        return route.train_path(path_id)
    except KeyError:
        raise RuntimeError(f"unknown path '{path_id}'")


def _parse_consist(install: msts.MSTSInstall, yd: typ.Union[str, typ.List[str]]) \
        -> typ.Sequence[tt.ConsistComponent]:
    if not isinstance(yd, list):
        return _parse_consist(install, [yd])

    consists = install.consists()
    def parse(subconsist: str) -> tt.ConsistComponent:
        split = subconsist.rsplit(maxsplit=1)
        reverse = len(split) == 2 and split[1].casefold() == '$reverse'
        con_id = split[0] if reverse else subconsist
        try:
            consist = install.consist(con_id)
        except KeyError:
            raise RuntimeError(f"unknown consist '{con_id}'")
        return tt.ConsistComponent(consist=consist, reverse=reverse)

    return [parse(item) for item in yd]


def _filter_trips(df: pd.DataFrame, filters: typ.Mapping[str, str]) -> pd.DataFrame:
    for prop, regex in filters.items():
        if prop not in ['route_id', 'service_id', 'trip_id', 'trip_headsign',
                        'trip_short_name', 'direction_id', 'block_id',
                        'shape_id', 'wheelchair_accessible', 'bikes_allowed']:
            raise KeyError(f'not a valid GTFS trip attribute: {prop}')
        df = df[df[prop].astype(str).str.match(regex)]
    return df


def _is_trip_start(feed: gk.feed.Feed, trip_id: _TripId, date: dt.date) -> bool:
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

    trip = feed.trips[feed.trips['trip_id'].astype(str) == trip_id].squeeze()
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


def _stop_times(feed: gk.feed.Feed, trip_id: _TripId,
                map_station: typ.Callable[[_TripId, _StopId], msts.Station]) \
        -> typ.Generator[_StopTime, None, None]:
    trip = (feed.stop_times[feed.stop_times['trip_id'] == trip_id]
        .sort_values(by='stop_sequence'))
    times = trip[['arrival_time', 'departure_time']].astype(str)
    stop_ids = trip['stop_id'].astype(str)

    def parse_stop(idx: int) -> typ.Optional[_StopTime]:
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

    def parse_time(s: str) -> typ.Tuple[int, dt.time]:
        match = re.search(r'(\d?\d)\s*:\s*([012345]?\d)\s*:\s*([012345]?\d)', s)
        if not match:
            raise ValueError(f'invalid GTFS time: {s}')
        hours = int(match.group(1))
        return (hours//24, dt.time(hour=hours%24,
                                   minute=int(match.group(2)),
                                   second=int(match.group(3))))

    return (st for st in (parse_stop(i) for i in range(len(trip)))
            if st is not None)


def _map_stations(route: msts.Route, feed: gk.feed.Feed) \
        -> typ.Mapping[_StopId, msts.Station]:
    @lru_cache(maxsize=64)
    def tokens(s: str) -> typ.Iterable[str]:
        return re.split('[ \t:;,-]+', s.casefold())

    word_frequency = Counter(
        chain(*(tokens(s_name) for s_name in route.station_names())))
    def similarity(a: str, b: str) -> float:
        ta = set(tokens(a))
        tb = set(tokens(b))
        intersect = set(ta & tb)
        difference = set.symmetric_difference(ta, tb)
        return -0.1*len(difference) \
            + sum(1/word_frequency[token] if token in word_frequency else 0.0
                  for token in intersect)

    geod = pp.Geod(ellps='WGS84')
    def dist_km(a: typ.Tuple[float, float], b: typ.Tuple[float, float]) -> float:
        lat_a, lon_a = a
        lat_b, lon_b = b
        _, _, dist = geod.inv(lon_a, lat_a, lon_b, lat_b)
        return dist/1000.0

    def map_station(stop: pd.Series) -> typ.Optional[str]:
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


def _name_trip(feed: gk.feed.Feed, trip_id: _TripId) -> str:
    trip = feed.trips[feed.trips['trip_id'] == trip_id].squeeze()
    route = feed.routes[feed.routes['route_id'] == trip['route_id']].squeeze()
    route_name = \
        route.get('route_long_name', '') or route.get('route_short_name', '')
    name = trip.get('trip_short_name', '') or trip_id
    headsign = trip.get('trip_headsign', '')
    if route_name:
        return f'{route_name} {name}'
    elif headsign:
        return f'{name} {headsign}'
    else:
        return name


_T = typ.TypeVar('_T')
_U = typ.TypeVar('_U')


def _strkeys(d: typ.Mapping[_T, _U]) -> typ.Mapping[str, _U]:
    return {str(k): v for k, v in d.items()}


def _reverse(d: typ.Mapping[_T, _U]) -> typ.Mapping[_U, typ.Iterable[_T]]:
    res = defaultdict(lambda: set())
    for k, v in d.items():
        res[v].add(k)
    return res


if __name__ == '__main__':
    freeze_support()
    main()

