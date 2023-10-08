# Before running this script, download the original dataset from https://www.garrickorchard.com/datasets/n-mnist.
# Create the directories 'input' and 'output' next to  this file.
# Move the unzipped 'Train' and 'Test' directories in the 'input' directory.
# The file  structure should be as follows.
#
# nmnist/
# ├─ input/
# │  ├─ Test/ (downloaded and unzipped)
# │  │  └─ ...
# │  └─ Train/ (downloaded and unzipped)
# │     └─ ...
# ├─ output (empty directory)
# └─ nmnist.py (this script)
#
# After compressing the dataset, run 'undr check-conformance output/nmnist'

import pathlib

import numpy
import undrdg

dirname = pathlib.Path(__file__).resolve().parent
type = undrdg.DvsType(width=34, height=34)


def read_nmnist_bin(path: pathlib.Path) -> numpy.ndarray:
    with open(path, "rb") as input:
        bytes = input.read()
    data = numpy.frombuffer(
        bytes[: (len(bytes) // 5) * 5],
        dtype=[("d0", "u1"), ("d1", "u1"), ("d2", "u1"), ("d3", "u1"), ("d4", "u1")],
    )
    events = numpy.zeros(len(data), dtype=type.dtype())
    events["t"] = numpy.bitwise_or(
        data["d4"].astype("<u8"),
        numpy.bitwise_or(
            numpy.left_shift(data["d3"].astype("<u8"), 8),
            numpy.left_shift(
                numpy.bitwise_and(data["d2"], 0b1111111).astype("<u8"), 16
            ),
        ),
    )
    events["x"] = data["d0"].astype("<u2")
    events["y"] = data["d1"].astype("<u2")
    events["p"] = numpy.right_shift(data["d2"], 7)
    if numpy.count_nonzero(numpy.diff(events["t"].astype("<i8")) < 0) > 0:  # type: ignore
        raise Exception("timestamp overflow")
    return events


def handle_task(task: undrdg.Task):
    events = read_nmnist_bin(task.source)
    with task.target_directory.create_file(
        type=type,
        name=task.target_name,
        metadata={
            "original_name": task.source.name,
            "scene": "screen",
            "sensor": "atis",
        },
    ) as file:
        file.write(events)
    print(
        f"{task.index + 1} / {task.total}",
        task.target_directory.path / task.target_name,
    )


undrdg.copy_tree(
    source=dirname / "input",
    target=dirname / "output" / "nmnist",
    rules=[
        undrdg.SkipName(".DS_Store"),
        undrdg.Rename("Test", "test"),
        undrdg.Rename("Train", "train"),
        undrdg.RenameExtension(".bin", ""),
    ],
    handle_task=handle_task,
)
