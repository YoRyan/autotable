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

```
$ pip3 install https://github.com/YoRyan/autotable
```

```
$ autotable --help
usage: autotable [-h] msts yaml

Generate Open Rails timetables from GTFS data.

positional arguments:
  msts        path to MSTS installation or mini-route
  yaml        path to timetable recipe file

optional arguments:
  -h, --help  show this help message and exit
```

### Recipe files

Timetable recipes are YAML files that filter trips from GTFS files and apply
properties and control commands.

```
route: SOME_ROUTE       # The name of the route's directory in ROUTES\.

date: 2020-01-01        # Filter from trips that overlap this date. Take care
                        # that your GTFS feeds are in service on this date.
gtfs:
    # A single timetable can source from multiple GTFS files.
    #
  - file: path/to/my/gtfs.zip           # Load a GTFS file from the filesystem...

    url: http://example.com/my/gtfs.zip # ...or retrieve an online GTFS file.
                                        # (Specify 'file' or 'url' but not both.)

    groups:
        # Groups apply a path, consist, and other properties to a selectable set
        # of GTFS trips.
        #
        # They are processed first-to-last and they override previously defined
        # properties, so you can add smaller groups to fine-tune previously
        # included trips.
        #
      - selection:
            # You filter GTFS trips using the attributes as defined in the spec:
            #   https://developers.google.com/transit/gtfs/reference#tripstxt
            #
            # Here, the key (before the :) is an attribute name, and the value
            # (after the :) is a regular expression to match attribute values.
            #
            # Multiple filter attributes are applied in an AND relationship.
            #
            trip_short_name: '^your regex here$'

        path: some path         # The filename of the trips' path, without the
                                # .pat extension.

        consist: some consist   # The filename of trips' consist, without the
                                # .con extension. (Currently, autotable does not
                                # support the extended consist syntax supported
                                # by Open Rails timetables.)

        start: -MM:SS           # Set the spawn time for this trip relative to
                                # the arrival time at its first on-route stop.
                                #
                                # Negative values push the time back, positive
                                # values move it forward (thus spawning a "late"
                                # train).
                                #
                                # You will want to adjust this based on the
                                # distance between the path's start node and its
                                # first stop.
                                #
                                # The default is 120 seconds before.

        note: ''                # Set control commands that apply for the entire
                                # trips' duration. The Open Rails manual suggests
                                # setting "$dec 2" or "$dec 3" for modern consists.

        dispose: ''             # Set the disposal commands that apply when the
                                # trips end.
    station_map:
        # autotable attempts to map GTFS stops to in-game platforms by first
        # filtering all platforms within a 10km radius (to account for
        # route-building inaccuracies) and then looking for significant words
        # that are common to both the GTFS and in-game names.
        #
        # Should this method fail, you can specify your own custom mappings here,
        # which will override the automatic ones. The key is the GTFS stop_id
        # attribute and the value is the MSTS/Open Rails station name.
        #
        # Specify a blank station name to denote a GTFS stop that explicitly has
        # no in-game equivalent, to resolve ambiguous cases where the same stop
        # apparently maps to multiple stations.
        #
        stop id: Station Name
```

For example recipes based on the routes that came with Microsoft Train Simulator,
see the [samples/](samples/) directory.
