# -*- coding: utf-8 -*-
import csv
import datetime as dt
import re
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import reduce

from more_itertools import pairwise

from autotable.mstsinstall import Consist, Route


@dataclass
class Trip:
    name: str
    stops: list
    path: Route.Path
    consist: list
    start_offset: int
    start_commands: str
    note_commands: str
    speed_commands: str
    delay_commands: str
    station_commands: dict
    dispose_commands: str

    def start_time(self):
        if len(self.stops) < 1:
            return None

        first_stop = self.stops[0].arrival
        return first_stop + dt.timedelta(seconds=self.start_offset)


Stop = namedtuple('Stop', ['station', 'comment', 'arrival', 'departure'])


@dataclass
class ConsistComponent:
    consist: Consist
    reverse: False

    def __str__(self):
        if re.search(r'[\+\$]', self.consist.id):
            if self.reverse:
                return f'<{self.consist.id}>$reverse'
            else:
                return f'<{self.consist.id}>'
        elif self.reverse:
            return f'{self.consist.id} $reverse'
        else:
            return self.consist.id


class SpeedUnit(Enum):
    MS = 1
    KPH = 2
    MPH = 3


@dataclass
class Timetable:
    name: str
    route: Route
    date: dt.date
    tz: dt.timezone
    trips: list
    station_commands: {}
    speed_unit: SpeedUnit

    def write_csv(self, fp):
        # csv settings per the May 2017 timetable document
        # http://www.elvastower.com/forums/index.php?/topic/30326-update-timetable-mode-signalling/
        writer = csv.writer(fp, delimiter='\t', quoting=csv.QUOTE_NONE)
        def writerow(*args): writer.writerow(args)

        def strftime(dt: dt.datetime) -> str:
            return dt.astimezone(self.tz).strftime('%H:%M')

        ordered_stations = _order_stations(
            self.trips, iter(self.route.stations().keys()))
        ordered_trips = self.trips

        writerow('', '', '#comment', *(trip.name for trip in ordered_trips))
        writerow('#comment', '', self.name)
        writerow('#path', '', '', *(trip.path.id for trip in ordered_trips))

        def consist_col(trip: Trip) -> str:
            return '+'.join(str(subconsist) for subconsist in trip.consist)
        writerow('#consist', '', '', *(consist_col(trip) for trip in ordered_trips))

        def start_col(trip: Trip) -> str:
            if trip.start_commands:
                return f'{strftime(trip.start_time())} {trip.start_commands}'
            else:
                return strftime(trip.start_time())
        writerow('#start', '', '', *(start_col(trip) for trip in ordered_trips))

        writerow('#note', '', '', *(trip.note_commands for trip in ordered_trips))

        speed_commands = (trip.speed_commands for trip in ordered_trips)
        if self.speed_unit == SpeedUnit.MS:
            writerow('#speed', '', '', *speed_commands)
        elif self.speed_unit == SpeedUnit.KPH:
            writerow('#speedkph', '', '', *speed_commands)
        elif self.speed_unit == SpeedUnit.MPH:
            writerow('#speedmph', '', '', *speed_commands)

        writerow('#restartdelay', '', '',
                 *(trip.delay_commands for trip in ordered_trips))

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
                yield f'{time} {commands}' if commands else time

        def station_comments(s_name: str):
            for i, _ in enumerate(ordered_trips):
                stop = stops_index.get((i, s_name), None)
                yield stop.comment if stop is not None else ''

        writerow()
        for s_name in ordered_stations:
            commands = self.station_commands.get(
                s_name, self.station_commands.get('', ''))
            writerow(s_name, commands, '', *station_stops(s_name))
            writerow('#comment', '', '', *station_comments(s_name))
        writerow()

        writerow('#dispose', '', '',
                 *(trip.dispose_commands for trip in ordered_trips))


def _order_stations(trips, stations):
    def add_trip(current_order, trip):
        def merge_in(order):
            return list(merge_inb(order))

        current_idx = {station: i for i, station in enumerate(current_order)}
        def merge_inb(order):
            ptr = 0
            for station in order:
                if station in current_idx:
                    yield from current_order[ptr:current_idx[station]]
                    ptr = max(ptr, current_idx[station] + 1)
                yield station
            yield from current_order[ptr:]

        def score(order):
            idx = {station: i for i, station in enumerate(order)}
            return sum(1 if idx[s1] < idx[s2] else 0 for s1, s2
                       in pairwise(filter((lambda s: s in idx), current_order)))

        trip_order = [stop.station for stop in trip.stops]
        return max(
            merge_in(trip_order), merge_in(list(reversed(trip_order))), key=score)

    return reduce(add_trip, trips, [])
