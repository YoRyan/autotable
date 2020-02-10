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
    def __init__(self, path: Path, encoding=ENCODING):
        df = ichild(path, f'{path.name}.trk')
        with open(df, encoding=ENCODING) as fp:
            d = kf.load(fp)
        desc = d['Tr_RouteFile']

        self.path = path
        self.id = desc['RouteID']
        self.name = desc['Name']
        self.description = desc['Description']
        self._filename = desc['FileName']

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

