from functools import lru_cache
from pathlib import Path

import kujufile as kf


ENCODING = 'utf-16'

class MSTSInstall:
    def __init__(self, path: Path, encoding=ENCODING):
        self.path = path
        self.routes = \
            [Route(child) for child in ichild(self.path, 'routes').iterdir()]
        self.consists = \
            [Consist(child) for child in ichild(ichild(self.path, 'trains'),
                                                'consists').iterdir()]

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
            self.player = desc['TrPathFlags'] & 0x20 == 0

    def __init__(self, path: Path, encoding=ENCODING):
        df = ichild(path, f'{path.name}.trk')
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
    def stations(self):
        df = ichild(self.path, f'{self._filename}.tit')
        with open(df, encoding=self._encoding) as fp:
            d = kf.load(fp)
        table = d['TrItemTable']

        return list(set(str(pi['Station']) for pi in table['PlatformItem']
                        if 'Station' in pi and pi['Station'] != ''))

    def paths(self):
        return [Route.Path(child, encoding=self._encoding)
                for child in ichild(self.path, 'paths').iterdir()
                if child.suffix.lower() == '.pat']

class Consist:
    def __init__(self, path: Path, encoding=ENCODING):
        with open(path, encoding=encoding) as fp:
            d = kf.load(fp)
        config = d['Train']['TrainCfg']

        self.path = path
        self.id = str(config.values()[0])
        self.name = str(config['Name']) if 'Name' in config else None

def ichild(path, name): return next(child for child in path.iterdir()
                                    if child.name.lower() == name.lower())

