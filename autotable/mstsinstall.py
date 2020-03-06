# -*- coding: utf-8 -*-
from concurrent.futures import as_completed, ProcessPoolExecutor
from collections import defaultdict, namedtuple
from functools import lru_cache
from pathlib import Path

import pyproj as pp

import autotable.kujufile as kf


ENCODING = 'utf-16'

class MSTSInstall:
    def __init__(self, path: Path, encoding=ENCODING):
        self.path = path
        route_dirs = (child for child in _ichild(self.path, 'routes').iterdir()
                      if child.is_dir())
        with ProcessPoolExecutor() as executor:
            loaded_routes = executor.map(Route, route_dirs)
        self.routes = {route.id.casefold(): route for route in loaded_routes}

    @lru_cache(maxsize=1)
    def consists(self) -> list:
        con_files = (child for child in _ichild(_ichild(self.path, 'trains'),
                                                'consists').iterdir()
                     if child.suffix.casefold() == '.con'.casefold())
        with ProcessPoolExecutor() as executor:
            consists = executor.map(Consist, con_files)
        return {consist.id.casefold(): consist for consist in consists}


class Route:
    class Path:
        def __init__(self, path: Path, encoding=ENCODING):
            with open(path, encoding=encoding) as fp:
                d = kf.load(fp)
            desc = d['TrackPath']

            self.path = path
            self.id = desc['TrPathName']
            self.name = desc['Name']
            self.start = desc['TrPathStart']
            self.end = desc['TrPathEnd']
            self.player = (desc['TrPathFlags'] & 0x20 == 0
                           if 'TrPathFlags' in desc else True)

    class PlatformItem:
        def __init__(self, data: kf.Object):
            self.name = data['PlatformName']
            self.station = data['Station']

            if 'TrItemRData' not in data:
                self.elevation_m = 0
                self.latlon = (0, 0)
                return

            rdata = data['TrItemRData'].values()
            self.elevation_m = rdata[1]

            # Conversion is abridged from Open Rails
            # (Orts.Common.WorldLatLon.ConvertWTC)
            radius_m = 6370997
            tile_m = 2048
            ul_y = 8673000
            ul_x = -20015000
            ns_offset = 16385
            ew_offset = -ns_offset
            y = ul_y - (ns_offset - rdata[4] - 1)*tile_m + rdata[2]
            x = ul_x + (rdata[3] - ew_offset - 1)*tile_m + rdata[0]
            goode = pp.CRS.from_proj4(f'+proj=igh +R={radius_m}')
            self.latlon = pp.transform(goode, pp.Proj('epsg:4326'), x, y)

    def __init__(self, path: Path, encoding=ENCODING):
        df = next(child for child in path.iterdir()
                  if child.suffix.casefold() == '.trk')
        with open(df, encoding=encoding) as fp:
            d = kf.load(fp)
        desc = d['Tr_RouteFile']

        self.path = path
        self.id = desc['RouteID']
        self.name = desc['Name']
        self.description = desc['Description']
        self._filename = desc['FileName']
        self._encoding = encoding

    def __hash__(self):
        return hash((self.path, self.id))

    @lru_cache(maxsize=1)
    def stations(self) -> dict:
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

    def station_names(self) -> iter: return self.stations().keys()

    @lru_cache(maxsize=1)
    def paths(self) -> list:
        route_paths = (child for child in _ichild(self.path, 'paths').iterdir()
                       if child.suffix.casefold() == '.pat'.casefold())
        with ProcessPoolExecutor() as executor:
            futures = []
            for path in route_paths:
                futures.append(
                    executor.submit(Route.Path, path, encoding=self._encoding))
            loaded_paths = [future.result() for future in as_completed(futures)]
        return {path.id.casefold(): path for path in loaded_paths}


class Consist:
    def __init__(self, path: Path, encoding=ENCODING):
        with open(path, encoding=encoding) as fp:
            d = kf.load(fp)
        config = d['Train']['TrainCfg']

        self.path = path
        self.id = str(config.values()[0])
        self.name = str(config['Name']) if 'Name' in config else None


def _ichild(path, name): return next(child for child in path.iterdir()
                                     if child.name.casefold() == name.casefold())

