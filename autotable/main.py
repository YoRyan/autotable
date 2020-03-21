# -*- coding: utf-8 -*-
import datetime as dt
import re
import typing as typ
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, tee
from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd # type: ignore
import pyproj as pp # type: ignore
import pytz
import yaml
from more_itertools import first, ilen, take
from timezonefinder import TimezoneFinder # type: ignore

import autotable.gtfs as gtfs
import autotable.mstsinstall as msts
import autotable.timetable as tt
from autotable import __version__


_MIN_STOPS = 1

_TzF = TimezoneFinder()


@dataclass
class _StopTime:
    station: msts.Station
    mapped_stop_id: gtfs.StopId
    arrival: gtfs.StopTime
    departure: gtfs.StopTime


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
    station_map: typ.Dict[gtfs.StopId, msts.Station]

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
            feed = gtfs.read_gtfs(feed_path)
        elif 'url' in yd:
            feed_path = yd['url']
            feed = gtfs.download_gtfs(feed_path)
        else:
            raise RuntimeError("'gtfs' block missing a 'file' or 'url'")
        ifeed = gtfs.IndexedFeed(feed)

        def validate_stop(stop_id: gtfs.StopId) -> None:
            if stop_id not in ifeed.stops.index:
                raise RuntimeError(
                    f"unknown stop ID '{stop_id}' for the gtfs feed {feed_path}")

        def stop_name(stop_id: gtfs.StopId) -> str:
            return ifeed.stops.at[stop_id, 'stop_name']

        # Read attributes from trip blocks.
        prelim_trips: typ.Mapping[gtfs.StopId, _TripConfig] = \
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
        trip_groups: typ.Dict[gtfs.TripId, int] = {}
        for i, group in enumerate(yd.get('groups', [])):
            rows = _filter_trips(feed.get_trips(), group.get('selection', {}))
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

                trip_groups[gtfs.TripId(trip_id)] = i

        # Map GTFS stops to station names.
        auto_map = _map_stations(route, ifeed)
        gtfs_map = _strkeys(yd.get('station_map', {}))
        def map_station(trip_id: gtfs.TripId, stop_id: gtfs.StopId) -> msts.Station:
            return prelim_trips[trip_id].station_map.get(
                stop_id, gtfs_map.get(stop_id, auto_map.get(stop_id, None)))

        # Add all Trips to Timetable.
        def finalize_trip(trip: _TripConfig, trip_id: gtfs.TripId) \
                -> typ.Optional[tt.Trip]:
            if not trip.path or not trip.consist:
                return None

            st1, st2, st3 = tee(_stop_times(ifeed, trip_id, map_station), 3)
            if ilen(take(_MIN_STOPS, st1)) < _MIN_STOPS:
                return None

            first_st = first(st2)
            start_dt = dt.datetime.combine(
                date + dt.timedelta(days=-first_st.arrival.days_elapsed),
                first_st.arrival.time)
            if not _is_trip_start(
                    ifeed, trip_id,
                    (start_dt + dt.timedelta(seconds=trip.start_offset)).date()):
                return None

            trip_row = ifeed.trips.loc[trip_id]
            route = ifeed.routes.loc[trip_row['route_id']]
            agency = ifeed.agency.loc[route['agency_id']]
            # Assume route and trip timezone are identical if the GTFS feed
            # doesn't specify one.
            trip_timezone = (_parse_timezone(agency['agency_timezone'], date)
                             if agency['agency_timezone'] else timezone)
            def make_stop(st: _StopTime) -> tt.Stop:
                start_date = start_dt.date()
                arrival_dt = dt.datetime.combine(
                    start_date + dt.timedelta(days=st.arrival.days_elapsed),
                    st.arrival.time,
                    tzinfo=trip_timezone)
                departure_dt = dt.datetime.combine(
                    start_date + dt.timedelta(days=st.departure.days_elapsed),
                    st.departure.time,
                    tzinfo=trip_timezone)
                return tt.Stop(
                    station=st.station,
                    comment=f'{st.mapped_stop_id} {stop_name(st.mapped_stop_id)}',
                    arrival=arrival_dt,
                    departure=departure_dt)

            return trip.finalize(_name_trip(ifeed, trip_id),
                                 [make_stop(st) for st in st3])

        grouped_trips = _reverse(trip_groups)
        for i in sorted(grouped_trips.keys()):
            trips = [trip for trip in (finalize_trip(prelim_trips[trip_id], trip_id)
                                       for trip_id in grouped_trips[i])
                     if trip is not None]
            trips.sort(key=tt.Trip.start_time)
            yield from trips

    blocks = yd.get('gtfs', None)
    if not isinstance(blocks, list):
        raise RuntimeError("'gtfs' not present or not a list of dictionaries")
    trips: typ.List[tt.Trip] = []
    for block in blocks:
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
        df = df[df[prop].astype(str).str.contains(regex)]
    return df


def _is_trip_start(ifeed: gtfs.IndexedFeed, trip_id: gtfs.TripId, date: dt.date) \
        -> bool:
    """Equivalent to gtfs_kit.trips.is_trip_active(), but with our own fixes."""
    trip = ifeed.trips.loc[trip_id]
    service_id = trip['service_id']
    if (ifeed.calendar_dates is not None
            and (service_id, date) in ifeed.calendar_dates.index):
        exception = ifeed.calendar_dates.at[(service_id, date), 'exception_type']
        if exception == 1:
            return True
        elif exception == 2:
            return False
        else:
            assert False
    elif ifeed.calendar is not None and service_id in ifeed.calendar.index:
        in_range = ifeed.calendar.at[service_id, 'start_date'] \
            <= date \
            <= ifeed.calendar.at[service_id, 'end_date']
        day_match = ifeed.calendar.at[service_id, 'weekdays'][date.weekday()]
        return in_range and day_match
    else:
        return False


def _stop_times(ifeed: gtfs.IndexedFeed, trip_id: gtfs.TripId,
                map_station: typ.Callable[[gtfs.TripId, gtfs.StopId], msts.Station]) \
        -> typ.Generator[_StopTime, None, None]:
    def parse_st(row: pd.Series) -> typ.Optional[_StopTime]:
        stop_id = row['stop_id']
        station = map_station(trip_id, stop_id)
        if not station:
            return None
        return _StopTime(station=station,
                         mapped_stop_id=stop_id,
                         arrival=row['arrival_time'],
                         departure=row['departure_time'])

    trip = ifeed.stop_times.loc[trip_id].sort_values(by='stop_sequence')
    parsed = trip.apply(parse_st, axis='columns').values.tolist()
    return (st for st in parsed if st is not None)


def _map_stations(route: msts.Route, ifeed: gtfs.IndexedFeed) \
        -> typ.Mapping[gtfs.StopId, msts.Station]:
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

    stops = ifeed.stops.copy()
    stops['_mapped_station'] = stops.apply(map_station, axis='columns')
    return stops.to_dict()['_mapped_station']


def _name_trip(ifeed: gtfs.IndexedFeed, trip_id: gtfs.TripId) -> str:
    trip = ifeed.trips.loc[trip_id]
    route = ifeed.routes.loc[trip['route_id']]

    def get(series: pd.Series, prop: str) -> typ.Any:
        val = series.get(prop, None)
        return None if pd.isna(val) else val

    route_name = get(route, 'route_long_name') or get(route, 'route_short_name')
    name = get(trip, 'trip_short_name') or trip_id
    headsign = get(trip, 'trip_headsign')
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

