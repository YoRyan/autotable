# -*- coding: utf-8 -*-
import csv
import datetime as dt
from collections import Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain, takewhile

from more_itertools import ilen, quantify, pairwise

from autotable.mstsinstall import Route


@dataclass
class Trip:
    name: str
    headsign: str
    route_name: str
    block: str
    stops: list
    path: Route.Path
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
        self.start_time = first_stop + dt.timedelta(seconds=self.start_offset)


Stop = namedtuple('TripStop', ['station', 'mapped_stop_id', 'mapped_stop_name',
                               'arrival', 'departure'])


class Timetable:

    def __init__(self, route: Route, date: dt.date, name: str,
                 tzinfo=dt.timezone.utc):
        self.route = route
        self.date = date
        self.tz = tzinfo
        self.name = name
        self.trips = []
        self.station_commands = {}

    def write_csv(self, fp):
        # csv settings per the May 2017 timetable document
        # http://www.elvastower.com/forums/index.php?/topic/30326-update-timetable-mode-signalling/
        writer = csv.writer(fp, delimiter='\t', quoting=csv.QUOTE_NONE)

        ordered_stations = _order_stations(
            self.trips, iter(self.route.stations().keys()))
        ordered_trips = self.trips

        def trip_name(trip: Trip) -> str:
            if trip.route_name:
                return f'{trip.route_name} {trip.name}'
            elif trip.headsign:
                return f'{trip.name} {trip.headsign}'
            else:
                return trip.name
        writer.writerow(chain(iter(('', '', '#comment')),
                              (trip_name(trip) for trip in ordered_trips)))

        writer.writerow(iter(('#comment', '', self.name)))
        writer.writerow(chain(iter(('#path', '', '')),
                              (trip.path.id for trip in ordered_trips)))

        def consist_col(trip: Trip) -> str:
            return '+'.join(str(subconsist) for subconsist in trip.consist)
        writer.writerow(chain(iter(('#consist', '', '')),
                              (consist_col(trip) for trip in ordered_trips)))

        def strftime(dt: dt.datetime) -> str:
            return dt.astimezone(self.tz).strftime('%H:%M')

        def start_col(trip: Trip) -> str:
            if trip.start_commands:
                return f'{strftime(trip.start_time)} {trip.start_commands}'
            else:
                return strftime(trip.start_time)
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
                {executor.submit(_station_order_cost, cand, sm_trips):
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
            length = ilen(takewhile(lambda s: s != common_stations[-1],
                                    iter(compare_order)))

        forward = (
            (quantify(trip_index[s1] < trip_index[s2]
                      for s1, s2 in pairwise(common_stations))
                 - quantify(trip_index[s1] > trip_index[s2]
                            for s1, s2 in pairwise(common_stations))),
            0,
            quantify(compare_index[s1] + 1 == compare_index[s2]
                     for s1, s2 in pairwise(common_stations)),
            (-quantify(trip_index[s1] + 1 != trip_index[s2]
                       for s1, s2 in pairwise(common_stations))
                 - length))
        backward = (
            (quantify(trip_index[s1] > trip_index[s2]
                      for s1, s2 in pairwise(common_stations))
                 - quantify(trip_index[s1] < trip_index[s2]
                            for s1, s2 in pairwise(common_stations))),
            -1,
            quantify(compare_index[s1] == compare_index[s2] + 1
                     for s1, s2 in pairwise(common_stations)),
            (-quantify(trip_index[s1] != trip_index[s2] + 1
                       for s1, s2 in pairwise(common_stations))
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
