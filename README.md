# autotable

autotable is a procedural timetable generator for the free
[Open Rails](http://openrails.org) train simulator. It uses
[GTFS](https://developers.google.com/transit) data to recreate real-life schedules.

As a timetable designer, you configure autotable through an easy-to-read YAML
recipe file that defines the consist, path, and other control commands for each
run. By sourcing data from GTFS feeds, autotable automates away the rote work of
copying and pasting (or manually entering) individual arrival and departure times.

autotable is a command-line tool written in Python 3. It uses
[GTFS Kit](https://github.com/mrcagney/gtfs_kit) to parse GTFS feeds, and an
internal reader to parse Microsoft Train Simulator/Open Rails data files.

### Quick start

As of February 2020, some of the PyPI dependencies will not build and install on
Windows. You can get prebuilt wheels (install them in the listed order) courtesy
of [Christoph Gohlke](https://www.lfd.uci.edu/~gohlke/pythonlibs/):

1. [GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
2. [Fiona](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona)
3. [rtree](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree)

Then, to install autotable:

```
>pip install git+https://github.com/YoRyan/autotable
```

```
>autotable --help
usage: autotable [-h] msts yaml

Generate Open Rails timetables from GTFS data.

positional arguments:
  msts        path to MSTS installation or mini-route
  yaml        path to timetable recipe file

optional arguments:
  -h, --help  show this help message and exit
```

### Recipe files

Timetable recipes are YAML files that select trips from GTFS files and apply
properties and control commands. Some example recipes are available in the
[samples/](samples/) directory.

```
route: SOME_ROUTE
date: 2020-01-01
gtfs:
  - file: path/to/my/gtfs.zip
    groups:
      - selection:
            trip_short_name: '^your regex here$'
        path: some path
        consist: some consist
        start: -MM:SS
        note: ''
        dispose: ''
    station_map:
        stop id: Station Name
    station_commands:
        station name: ''
```

(Unfortunately, the Open Rails manual does not yet document all available
commands. Refer to the May 2017 timetable
[design document](http://www.elvastower.com/forums/index.php?/topic/30326-update-timetable-mode-signalling/).)

Recipes should be YAML dictionaries with the following keys:

#### route

The name of the route's directory in ROUTES\\.

#### date

Select trips that overlap this date. Take care that your GTFS feeds are in
service on this date.

#### gtfs

A list of dictionaries representing the GTFS sources and their trips. A single
timetable can source from multiple GTFS files. Each gtfs block must specify a
`file` or `url` but not both.

##### file

Load a GTFS file from the local filesystem. The path is relative to the current
directory.

##### url

Load a GTFS file from the Internet. Must be a full HTTP or HTTPS URL.

##### groups

A list of dictionaries representing groups of trips. Groups apply a path,
consist, and other attributes to a particular subset of GTFS trips.

Groups are processed first-to-last and can override previously defined
attributes, so you can add smaller groups to fine-tune previously
included trips.

Trips will not be written to the timetable unless assigned both a consist
and path. Of course, they must also make at least one stop at a station
represented by the route.

###### selection

A dictionary that selects trips by their attributes as defined in
[the GTFS spec](https://developers.google.com/transit/gtfs/reference#tripstxt).
Each key represents an attribute name, and the corresponding value
represents a regular expression to match attribute values.

Multiple filtered attributes are applied in an AND relationship.

###### path

The filename of the trips' path, without the .pat extension.

###### consist

The filename of trips' consist, without the .con extension.

(Currently, autotable does not support the extended consist syntax supported
by Open Rails timetables.)

###### start

*Default: 120 seconds before*

Set trip spawn times relative to their arrival times at their first on-route
stops.

Negative values push the start time back, while positive values move it forward
(thus spawning a "late" train).

You will want to adjust this based on the distance between the path's start
node and its first stop.

###### note

Set *train* commands. The Open Rails manual
[suggests](https://open-rails.readthedocs.io/en/stable/timetable.html#special-rows)
using `$dec=2` or `$dec=3` for modern equipment.

###### speed_mps

Set *speed* commands in m/s units.

###### speed_kph

Set *speed* commands in km/h units.

###### speed_mph

Set *speed* commands in mi/h units.

###### dispose

Set the *dispose* commands that apply when the trips terminate at their last
represented stations.

###### station_commands

A dictionary that maps in-game station names to *station stop* commands.

###### station_map

This is equivalent to the `station_map` field of the `gtfs` block (below),
except that it applies only to trips selected by this `groups` block.

##### station_map

A dictionary that maps GTFS `stop_id`'s to their corresponding in-game station
names.

autotable tries to build this automatically by first filtering all platforms
within a 10km radius (to account for route-building inaccuracies) and then
looking for words that are common to both the GTFS and in-game names. Usually,
this heuristic gets it right, but you can fine-tune the results by adding
mappings here, which will override the automatic ones.

Specify a blank station name to denote a GTFS stop that explicitly has no
in-game equivalent, to resolve ambiguous cases where the same stop apparently
maps to multiple stations.

#### station_commands

A dictionary that maps in-game station names to *station* commands.

The special empty key `""` applies to all stations that do not have commands
specifically defined for themselves.

(Protip: Route builders often forget to change the minimum platform wait time
from the MSTS-default 3 minutes, so use the `$stoptime=s` command to specify
your own.)
