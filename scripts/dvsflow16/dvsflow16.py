# Before running this script, download the original dataset from http://sensors.ini.uzh.ch/databases.html.
# Create the directories 'input' and 'output' next to  this file.
# Move the downloaded 'real samples' and 'synthesized samples' directories in the 'input' directory.
# The file  structure should be as follows.
#
# dvsflow16/
# ├─ input/
# │  ├─ real samples/ (downloaded)
# │  │  └─ ...
# │  └─ synthesized samples/ (downloaded)
# │     └─ ...
# ├─ output (empty directory)
# └─ dvsflow16.py (this script)
#
# After compressing the dataset, run 'undr check-conformance output/dvsflow16'

import pathlib
import shutil

import undrdg

dirname = pathlib.Path(__file__).resolve().parent

dvs_type = undrdg.DvsType(width=240, height=180)
aps_type = undrdg.ApsType(width=240, height=180)
imu_type = undrdg.ImuType()


def handle_task(task: undrdg.Task):
    name, date = undrdg.stem_and_date(
        stem=task.source.stem,
        trim_prefixes=["DAVIS240C-"],
        trim_suffixes=[],
    )
    for prefix in ("00000075_0_", "84010015_0_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    metadata = {
        "original_name": task.source.name,
    }
    if date is not None:
        metadata["date"] = date
    if task.source.suffix == ".aedat":
        aerdat2 = undrdg.Aerdat2.read(task.source)
        if aerdat2.header is not None:
            with task.target_directory.create_other_file(
                name=f"{name}.header", metadata=metadata
            ) as file:
                file.write(aerdat2.header)
        metadata = metadata.copy()
        if task.source.relative_to(task.source_root).parts[0] == "real samples":
            metadata["scene"] = "real"
        else:
            metadata["scene"] = "synthetic"
        metadata["sensor"] = "davis240c"
        if aerdat2.dvs is not None:
            with task.target_directory.create_file(
                type=dvs_type,
                name=name,
                metadata=metadata,
            ) as file:
                file.write(aerdat2.dvs)
        if aerdat2.aps is not None:
            with task.target_directory.create_file(
                type=aps_type,
                name=name,
                metadata=metadata,
            ) as file:
                file.write(aerdat2.aps)
        if aerdat2.imu is not None:
            with task.target_directory.create_file(
                type=imu_type,
                name=name,
                metadata=metadata,
            ) as file:
                file.write(aerdat2.imu)
        if aerdat2.aps_corrupted:
            print(f"{task.target_directory.path / name}: APS corrupted")
        if aerdat2.imu_corrupted:
            print(f"{task.target_directory.path / name}: IMU corrupted")
    else:
        with task.target_directory.create_other_file(
            name=f"{name}{task.source.suffix}", metadata=metadata
        ) as file:
            for chunk in undrdg.read_chunks(task.source):
                file.write(chunk)
    print(
        f"{task.index + 1} / {task.total}",
        task.target_directory.path / name,
    )


for name in ("gtRotatingBar.mat", "gtTranslatingSquare.mat"):
    shutil.copy2(
        (dirname / "input" / "synthesized samples" / "ground truth" / name),
        (dirname / "input" / "synthesized samples" / name),
    )

undrdg.copy_tree(
    source=dirname / "input",
    target=dirname / "output" / "dvsflow16",
    doi="10.3389/fnins.2016.00176",
    rules=[
        undrdg.SkipName(".DS_Store"),
        undrdg.SkipName("README.txt"),
        undrdg.SkipName("checksums.md5"),
        undrdg.SkipName("ground truth"),
        undrdg.Rename("real samples", "real"),
        undrdg.Rename("synthesized samples", "synthesized"),
        undrdg.Rename("real samples/DAVIS rectangle", "davis_rectangle"),
        undrdg.Rename("real samples/IMU_APS", "imu_aps"),
        undrdg.Rename("real samples/IMU_DVS", "imu_dvs"),
        undrdg.Rename("real samples/noIMU", "no_imu"),
        undrdg.RenameExtension(".bin", ""),
    ],
    handle_task=handle_task,
)
