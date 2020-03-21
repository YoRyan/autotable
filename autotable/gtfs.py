# -*- coding: utf-8 -*-
import datetime as dt
import re
import typing as typ
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import gtfs_kit as gk # type: ignore
import pandas as pd # type: ignore
from requests import get


_GTFS_UNITS = 'm'


AgencyId = str
StopId = str
RouteId = str
TripId = str
ServiceId = str


class IndexedFeed:
    def __init__(self, feed: gk.feed.Feed):
        self._feed: gk.feed.Feed = feed

        stops = feed.get_stops()
        stops['stop_id'] = stops['stop_id'].astype(StopId)
        self.stops: pd.DataFrame = stops.set_index('stop_id')

        trips = feed.get_trips()
        trips['trip_id'] = trips['trip_id'].astype(TripId)
        trips['service_id'] = trips['service_id'].astype(ServiceId)
        trips['route_id'] = trips['route_id'].astype(RouteId)
        self.trips: pd.DataFrame = trips.set_index('trip_id')

        agency = feed.agency.copy()
        agency['agency_id'] = agency['agency_id'].astype(AgencyId)
        self.agency: pd.DataFrame = agency.set_index('agency_id')

        routes = feed.get_routes()
        routes['route_id'] = routes['route_id'].astype(RouteId)
        routes['agency_id'] = routes['agency_id'].astype(AgencyId)
        self.routes: pd.DataFrame = routes.set_index('route_id')

        st = feed.get_stop_times()
        st['trip_id'] = st['trip_id'].astype(TripId)
        st['stop_id'] = st['stop_id'].astype(StopId)
        st['arrival_time'] = st['arrival_time'].apply(_strptime)
        st['departure_time'] = st['departure_time'].apply(_strptime)
        self.stop_times: pd.DataFrame = st.set_index('trip_id')

        self.calendar_dates: typ.Optional[pd.DataFrame] = None
        if feed.calendar_dates is not None:
            cd = feed.calendar_dates.copy()
            cd['service_id'] = cd['service_id'].astype(ServiceId)
            cd['date'] = cd['date'].astype(str).apply(_strpdate)
            cd['exception_type'] = cd['exception_type'].astype(int)
            self.calendar_dates = cd.set_index(['service_id', 'date'])

        self.calendar: typ.Optional[pd.DataFrame] = None
        if feed.calendar is not None:
            cal = feed.calendar.copy()
            cal['service_id'] = cal['service_id'].astype(ServiceId)
            cal['start_date'] = cal['start_date'].astype(str).apply(_strpdate)
            cal['end_date'] = cal['end_date'].astype(str).apply(_strpdate)
            cal['weekdays'] = cal.apply(_weekdays, axis='columns')
            self.calendar = cal.set_index('service_id')


@dataclass
class StopTime:
    days_elapsed: int
    time: dt.time


@lru_cache(maxsize=8)
def read_gtfs(path: Path) -> gk.feed.Feed:
    return gk.read_feed(path, dist_units=_GTFS_UNITS)


@lru_cache(maxsize=8)
def download_gtfs(url: str) -> gk.feed.Feed:
    tf = NamedTemporaryFile(delete=False)
    with get(url, stream=True) as req:
        for chunk in req.iter_content(chunk_size=128):
            tf.write(chunk)
    tf.close()
    gtfs = gk.read_feed(tf.name, dist_units=_GTFS_UNITS)
    Path(tf.name).unlink()
    return gtfs


def _strptime(s: str) -> StopTime:
    match = re.search(r'(\d?\d)\s*:\s*([012345]?\d)\s*:\s*([012345]?\d)', s)
    if not match:
        raise ValueError(f'invalid GTFS time: {s}')
    hours = int(match.group(1))
    return StopTime(days_elapsed=hours//24,
                    time=dt.time(hour=hours%24,
                                 minute=int(match.group(2)),
                                 second=int(match.group(3))))


def _strpdate(s: str) -> dt.date:
    return dt.datetime.strptime(s.strip(), '%Y%m%d').date()


def _weekdays(row: pd.Series) -> typ.Tuple[bool, ...]:
    # In Python, weeks start on Mondays.
    week = ('monday', 'tuesday', 'wednesday', 'thursday',
            'friday', 'saturday', 'sunday')
    return tuple(bool(row[day]) for day in week)
