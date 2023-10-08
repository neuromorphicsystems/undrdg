# Before running this script, download the original dataset from http://sensors.ini.uzh.ch/databases.html.
# Create the directories 'input' and 'output' next to  this file.
# Move the unzipped 'DVS09 - DVS128 sample data' directory in the 'input' directory.
#
# dvs09/
# ├─ input/
# │  └─ DVS09 - DVS128 sample data/ (downloaded and unzipped)
# │     └─ ...
# ├─ output (empty directory)
# └─ dvs09.py (this script)
#
# After compressing the dataset, run 'undr check-conformance output/dvs09'

import dataclasses
import pathlib
import typing

import numpy
import matplotlib.pyplot
import undrdg

matplotlib.use("agg")

dirname = pathlib.Path(__file__).resolve().parent
type = undrdg.DvsType(width=128, height=128)

directory = undrdg.Directory(path=dirname / "output" / "dvs09")


@dataclasses.dataclass
class Properties:
    name: str
    timestamp: str
    format: typing.Literal["aerdat1", "aerdat2"]
    timestamp_fix: typing.Optional[typing.Literal["add", "sort"]]


name_to_properties: dict[str, Properties] = {
    "Tmpdiff128-2006-02-10T14-22-35-0800-0 hand for orientation.dat": Properties(
        name="hand_for_orientation",
        timestamp="2006-02-10T14:22:35-08:00",
        format="aerdat1",
        timestamp_fix="add",
    ),
    "Tmpdiff128-2006-02-14T07-45-15-0800-0 walk to kripa.dat": Properties(
        name="walk_to_kripa",
        timestamp="2006-02-14T07:45:15-08:00",
        format="aerdat1",
        timestamp_fix=None,
    ),
    "Tmpdiff128-2006-02-23T12-48-34+0100-0 patrick juggling.dat": Properties(
        name="patrick_juggling",
        timestamp="2006-02-23T12:48:34+01:00",
        format="aerdat1",
        timestamp_fix=None,
    ),
    "Tmpdiff128-2006-06-17T10-48-06+0200-0 vogelsang saturday monring #2.dat": Properties(
        name="vogelsang_saturday_morning",
        timestamp="2006-06-17T10:48:06+02:00",
        format="aerdat1",
        timestamp_fix=None,
    ),
    "Tmpdiff128-2006-08-24T15-19-36+0200-0 patrick eye sharp.dat": Properties(
        name="patrick_eye_sharp",
        timestamp="2006-08-24T15:19:36+02:00",
        format="aerdat1",
        timestamp_fix=None,
    ),
    "Tmpdiff128-2006-09-04T08-51-54+0200-0 mouse 3p5d bg filtered.dat": Properties(
        name="mouse_3p5d_bg_filtered",
        timestamp="2006-09-04T08:51:54+02:00",
        format="aerdat1",
        timestamp_fix="add",
    ),
    "Tmpdiff128-2007-02-28T15-08-15-0800-0 3 flies 2m 1f.dat": Properties(
        name="flies_2m_1f",
        timestamp="2007-02-28T15:08:15-08:00",
        format="aerdat1",
        timestamp_fix="add",
    ),
    "Tmpdiff128-2008-04-09T15-38-53+0200-0fastDot.dat": Properties(
        name="fast_dot",
        timestamp="2008-04-09T15:38:53+02:00",
        format="aerdat2",
        timestamp_fix=None,
    ),
    "events-2005-12-28T11-14-28-0800 drive SC postoffice.dat": Properties(
        name="drive_sc_postoffice",
        timestamp="2005-12-28T11:14:28-08:00",
        format="aerdat1",
        timestamp_fix=None,
    ),
    "events-2006-01-23T21-49-01+0100 ori dir stimulus.dat": Properties(
        name="ori_dir_stimulus",
        timestamp="2006-01-23T21:49:01+01:00",
        format="aerdat1",
        timestamp_fix=None,
    ),
    "events20051219T171855 driving pasa freeway.mat.dat": Properties(
        name="driving_pasa_freeway",
        timestamp="2005-12-19T17:18:55-08:00",
        format="aerdat1",
        timestamp_fix=None,
    ),
    "events20051221T014416 freeway.mat.dat": Properties(
        name="freeway",
        timestamp="2005-12-21T01:44:16-08:00",
        format="aerdat1",
        timestamp_fix="sort",
    ),
}


(dirname / "output" / "figures").mkdir(exist_ok=True)


def plot_timestamps(timestamps: numpy.ndarray, path: pathlib.Path):
    timestamp_resets = numpy.argwhere(numpy.diff(timestamps.astype("i8")) < 0)[:, 0]
    matplotlib.pyplot.figure(figsize=(10, 6))
    for index in timestamp_resets:
        matplotlib.pyplot.axvline(x=index + 1, color="r")
    matplotlib.pyplot.plot(numpy.arange(0, len(timestamps)), timestamps)
    matplotlib.pyplot.xlabel("Event index")
    matplotlib.pyplot.ylabel("Event timestamp (µs)")
    matplotlib.pyplot.savefig(str(path))
    matplotlib.pyplot.close()


for path in sorted((dirname / "input" / "DVS09 - DVS128 sample data").iterdir()):
    if path.is_file() and path.suffix == ".dat":
        print(path)
        properties = name_to_properties[path.name]
        if properties.format == "aerdat1":
            aerdat1 = undrdg.parsers.read_aerdat1(path)
            assert aerdat1.dvs is not None
            events = aerdat1.dvs
        else:
            aerdat2 = undrdg.parsers.read_aerdat2(path)
            assert aerdat2.dvs is not None
            assert aerdat2.aps is None
            assert aerdat2.imu is None
            events = aerdat2.dvs
        timestamp_resets = numpy.argwhere(numpy.diff(events["t"].astype("i8")) < 0)[
            :, 0
        ]
        if len(timestamp_resets) > 0:
            plot_timestamps(
                timestamps=events["t"],
                path=dirname / "output" / "figures" / f"{properties.name}.png",
            )
        if properties.timestamp_fix is None:
            assert len(timestamp_resets) == 0
        elif properties.timestamp_fix == "add":
            for index in reversed(timestamp_resets):
                events["t"][index + 1 :] += events["t"][index]
            plot_timestamps(
                timestamps=events["t"],
                path=dirname
                / "output"
                / "figures"
                / f"{properties.name}_corrected.png",
            )
        else:
            events = numpy.sort(events, order=["t"], kind="stable")
            plot_timestamps(
                timestamps=events["t"],
                path=dirname
                / "output"
                / "figures"
                / f"{properties.name}_corrected.png",
            )
        with directory.create_file(
            type=type,
            name=properties.name,
            metadata={
                "date": properties.timestamp,
                "original_name": path.name,
                "scene": "real",
                "sensor": "dvs128",
            },
        ) as file:
            file.write(events)
