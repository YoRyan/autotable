# -*- coding: utf-8 -*-
import typing as typ
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pyproj as pp # type: ignore
from more_itertools import one

import autotable.kujufile as kf


ENCODING = 'utf-16'


class Ident(str):
    def __eq__(self, other) -> bool:
        t = type(other)
        return (self.casefold() == other.casefold()
                if t == Ident or t == str else False)

    def __hash__(self) -> int: return hash(self.casefold())


Station = str


class Route:
    class TrainPath:
        def __init__(self, path: Path, encoding=ENCODING):
            with open(path, encoding=encoding) as fp:
                d = kf.load(fp)
            desc = d['TrackPath']

            self.path: Path = path
            self.id: Ident = Ident(desc['TrPathName'])
            self.name: typ.Optional[str] = desc.get('Name', None)
            self.start: typ.Optional[str] = desc.get('TrPathStart', None)
            self.end: typ.Optional[str] = desc.get('TrPathEnd', None)
            self.player: bool = (desc['TrPathFlags'] & 0x20 == 0
                                 if 'TrPathFlags' in desc else True)

    class PlatformItem:
        def __init__(self, data: kf.Object):
            self.name: typ.Optional[str] = data.get('PlatformName', None)
            self.station: Station = str(data['Station'])
            self.elevation_m: float = 0
            self.latlon: typ.Tuple[float, float] = (0, 0)

            if ('TrItemRData' not in data
                    or not isinstance(data['TrItemRData'], kf.Object)):
                return

            ew, elev, ns, tile_x, tile_z = data['TrItemRData'].values()
            if (not isinstance(tile_x, int)
                    or not isinstance(tile_z, int)
                    or not isinstance(ew, float)
                    or not isinstance(ns, float)
                    or not isinstance(elev, float)):
                return

            self.elevation_m = elev
            self.latlon = _latlon(tile_x, tile_z, ew, ns)

    def __init__(self, path: Path, encoding=ENCODING):
        df = one(_echild(path, 'trk'))
        with open(df, encoding=encoding) as fp:
            d = kf.load(fp)
        desc = d['Tr_RouteFile']

        self.path: Path = path
        self.id: Ident = Ident(desc['RouteID'])
        self.name: typ.Optional[str] = desc.get('Name', None)
        self.description: typ.Optional[str] = desc.get('Description', None)

        tile_x, tile_z, ew, ns = desc['RouteStart'].values()
        self.latlon: typ.Tuple[float, float] = _latlon(tile_x, tile_z, ew, ns)

        self._filename: str = desc['FileName']
        self._encoding = encoding

    def __hash__(self): return hash((self.path, self.id))

    @lru_cache(maxsize=1)
    def stations(self) -> typ.Mapping[Station, typ.Iterable[PlatformItem]]:
        df = _ichild(self.path, f'{self._filename}.tit')
        with open(df, encoding=self._encoding) as fp:
            d = kf.load(fp)
        table = d['TrItemTable']

        with ProcessPoolExecutor() as executor:
            platforms = executor.map(Route.PlatformItem, table['PlatformItem'])
        res = defaultdict(list)
        for platform in platforms:
            res[platform.station].append(platform)
        return res

    def station_names(self) -> typ.Iterable[Station]: return self.stations().keys()

    @lru_cache(maxsize=1)
    def train_paths(self) -> typ.Mapping[Ident, TrainPath]:
        route_paths = _echild(_ichild(self.path, 'paths'), 'pat')
        with ProcessPoolExecutor() as executor:
            futures = []
            for path in route_paths:
                futures.append(
                    executor.submit(Route.TrainPath, path, encoding=self._encoding))
            loaded_paths = [future.result() for future in as_completed(futures)]
        return {path.id: path for path in loaded_paths}

    def train_path(self, id: str) -> TrainPath:
        return self.train_paths()[Ident(id)]


class Consist:
    def __init__(self, path: Path, encoding=ENCODING):
        with open(path, encoding=encoding) as fp:
            d = kf.load(fp)
        config = d['Train']['TrainCfg']

        self.path: Path = path
        self.id: Ident = Ident(config.values()[0])
        self.name: typ.Optional[str] = config.get('Name', None)


class MSTSInstall:
    def __init__(self, path: Path, encoding=ENCODING):
        self.path: Path = path

    @lru_cache(maxsize=1)
    def routes(self) -> typ.Mapping[Ident, Route]:
        route_dirs = (child for child in _ichild(self.path, 'routes').iterdir()
                      if child.is_dir())
        with ProcessPoolExecutor() as executor:
            return {route.id: route for route in executor.map(Route, route_dirs)}

    def route(self, id: str) -> Route:
        return self.routes()[Ident(id)]

    @lru_cache(maxsize=1)
    def consists(self) -> typ.Mapping[Ident, Consist]:
        con_files = \
            _echild(_ichild(_ichild(self.path, 'trains'), 'consists'), 'con')
        with ProcessPoolExecutor() as executor:
            return {consist.id: consist
                    for consist in executor.map(Consist, con_files)}

    def consist(self, id: str) -> Consist:
        return self.consists()[Ident(id)]


def _ichild(path: Path, name: str) -> Path:
    matches = (child for child in path.iterdir()
               if child.name.casefold() == name.casefold())
    return one(matches, too_short=FileNotFoundError, too_long=LookupError)


def _echild(path: Path, extension: str) -> typ.Iterable[Path]:
    suffix = f'.{extension.casefold()}'
    return (child for child in path.iterdir() if child.suffix.casefold() == suffix)


def _latlon(tile_x: int, tile_z: int, ew: float, ns: float) \
        -> typ.Tuple[float, float]:
    # Conversion is abridged from Open Rails (Orts.Common.WorldLatLon.ConvertWTC)
    # https://github.com/openrails/openrails/blob/master/Source/Orts.Simulation/Common/WorldLatLon.cs
    radius_m = 6370997
    tile_m = 2048
    ul_y = 8673000
    ul_x = -20015000
    ns_offset = 16385
    ew_offset = -ns_offset
    y = ul_y - (ns_offset - tile_z - 1)*tile_m + ns
    x = ul_x + (tile_x - ew_offset - 1)*tile_m + ew
    goode = pp.CRS.from_proj4(f'+proj=igh +R={radius_m}')
    return pp.transform(goode, pp.Proj('epsg:4326'), x, y)
