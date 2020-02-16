# -*- coding: utf-8 -*-
from collections import defaultdict, namedtuple
from functools import lru_cache
from pathlib import Path

import pyproj as pp

import autotable.kujufile as kf


ENCODING = 'utf-16'

class MSTSInstall:
    def __init__(self, path: Path, encoding=ENCODING):
        self.path = path
        self.routes = \
            [Route(child) for child in _ichild(self.path, 'routes').iterdir()]

    @lru_cache(maxsize=1)
    def consists(self) -> list:
        return [Consist(child) for child
                in _ichild(_ichild(self.path, 'trains'), 'consists').iterdir()]


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
        df = _ichild(path, f'{path.name}.trk')
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

        platforms = [Route.PlatformItem(pi) for pi in table['PlatformItem']]
        res = defaultdict(list)
        for platform in platforms:
            res[platform.station].append(platform)
        return res

    @lru_cache(maxsize=1)
    def paths(self) -> list:
        return [Route.Path(child, encoding=self._encoding)
                for child in _ichild(self.path, 'paths').iterdir()
                if child.suffix.lower() == '.pat']


class Consist:
    def __init__(self, path: Path, encoding=ENCODING):
        with open(path, encoding=encoding) as fp:
            d = kf.load(fp)
        config = d['Train']['TrainCfg']

        self.path = path
        self.id = str(config.values()[0])
        self.name = str(config['Name']) if 'Name' in config else None


def _ichild(path, name): return next(child for child in path.iterdir()
                                     if child.name.lower() == name.lower())

