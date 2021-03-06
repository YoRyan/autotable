from cx_Freeze import setup, Executable

from autotable import __version__


# Dependencies are automatically detected, but it might need
# fine tuning.
buildOptions = dict(packages = ['pyproj.datadir', 'pkg_resources.py2_warn'],
                    excludes = ['numpy.fft'],
                    # https://github.com/anthony-tuininga/cx_Freeze/issues/278#issuecomment-542316877
                    include_files = ['C:\\Windows\\System32\\vcruntime140.dll'])

base = 'Console'

executables = [
    Executable('autotable/main.py', base=base, targetName="autotable")
]

setup(name='autotable',
      version = __version__,
      description = 'An Open Rails timetable generator that uses GTFS data',
      options = dict(build_exe = buildOptions),
      executables = executables)
