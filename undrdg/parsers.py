import dataclasses
import pathlib
import typing

import undr
import numpy


class IncompleteHeader(Exception):
    def __init__(self):
        super().__init__("EOF reached while reading the header")


def read_aerdat_header(data: bytes) -> tuple[str, bytes]:
    index = 0
    while True:
        if index >= len(data):
            raise IncompleteHeader()
        if data[index] == 35:  # 35 is '#'
            line_end = data.find(b"\n", index + 1)
            if line_end == -1:
                break
            if not data[index:line_end].isascii():
                break
            index = line_end + 1
        else:
            break
    return (data[:index].decode(), data[index:])


@dataclasses.dataclass
class Aerdat1:
    header: typing.Optional[str]
    dvs: typing.Optional[numpy.ndarray]


def read_aerdat1(path: pathlib.Path) -> Aerdat1:
    aerdat1 = Aerdat1(
        header=None,
        dvs=None,
    )
    with open(path, "rb") as input:
        data = input.read()
    header, data = read_aerdat_header(data)
    if len(header) > 0:
        aerdat1.header = header
    data = data[: (len(data) // 6) * 6]
    if len(data) > 0:
        parsed_data = numpy.frombuffer(
            data, dtype=[("d0", "u1"), ("d1", "u1"), ("t", ">u4")]
        )
        aerdat1.dvs = numpy.zeros(len(parsed_data), dtype=undr.raw.DVS_DTYPE)
        aerdat1.dvs["t"] = parsed_data["t"]
        aerdat1.dvs["x"] = 127 - numpy.right_shift(parsed_data["d1"], 1)
        aerdat1.dvs["y"] = numpy.bitwise_and(parsed_data["d0"], 0x7F)
        aerdat1.dvs["p"] = numpy.bitwise_or(
            numpy.right_shift(numpy.bitwise_and(parsed_data["d0"], 0b10000000), 6),
            numpy.bitwise_and(parsed_data["d1"], 0b1),
        )
    return aerdat1


@dataclasses.dataclass
class Aerdat2:
    header: typing.Optional[str]
    dvs: typing.Optional[numpy.ndarray]
    aps: typing.Optional[numpy.ndarray]
    imu: typing.Optional[numpy.ndarray]
    aps_corrupted: bool
    imu_corrupted: bool

    @staticmethod
    def extract_x(parsed_data: numpy.ndarray) -> numpy.ndarray:
        return numpy.bitwise_or(
            numpy.left_shift(
                numpy.bitwise_and(parsed_data["d1"].astype("<u2"), 0b111111), 4
            ),
            numpy.right_shift(parsed_data["d2"].astype("<u2"), 4),
        )

    @staticmethod
    def extract_y(parsed_data: numpy.ndarray) -> numpy.ndarray:
        return numpy.bitwise_or(
            numpy.left_shift(
                numpy.bitwise_and(parsed_data["d0"].astype("<u2"), 0b1111111), 2
            ),
            numpy.right_shift(parsed_data["d1"].astype("<u2"), 6),
        )


def read_aerdat2(path: pathlib.Path) -> Aerdat2:
    aedat2 = Aerdat2(
        header=None,
        dvs=None,
        aps=None,
        imu=None,
        aps_corrupted=False,
        imu_corrupted=False,
    )
    with open(path, "rb") as input:
        data = input.read()
    header, data = read_aerdat_header(data)
    if len(header) > 0:
        aedat2.header = header
    data = data[: (len(data) // 8) * 8]
    parsed_data = numpy.frombuffer(
        data,
        dtype=[("d0", "u1"), ("d1", "u1"), ("d2", "u1"), ("d3", "u1"), ("t", ">u4")],
    )
    if numpy.count_nonzero(numpy.diff(parsed_data["t"].astype(">i8")) < 0) > 0:
        parsed_data = numpy.sort(parsed_data, order=["t"], kind="stable")
    dvs_mask = numpy.right_shift(parsed_data["d0"], 7) == 0
    aps_mask = numpy.logical_and(
        numpy.logical_not(dvs_mask),
        numpy.bitwise_and(parsed_data["d2"], 0b1100) != 0b1100,
    )
    imu_mask = numpy.logical_and(
        numpy.logical_not(dvs_mask), numpy.logical_not(aps_mask)
    )
    if numpy.count_nonzero(dvs_mask) > 0:
        dvs_data = parsed_data[dvs_mask]
        events = numpy.zeros(len(dvs_data), dtype=undr.raw.DVS_DTYPE)
        events["t"] = dvs_data["t"]
        events["x"] = Aerdat2.extract_x(dvs_data)
        events["y"] = Aerdat2.extract_y(dvs_data)
        events["p"] = numpy.bitwise_or(
            numpy.right_shift(
                numpy.bitwise_and(dvs_data["d2"].astype("<u2"), 0b1000), 3
            ),
            numpy.right_shift(
                numpy.bitwise_and(dvs_data["d2"].astype("<u2"), 0b100), 1
            ),
        )
        aedat2.dvs = events
    if numpy.count_nonzero(aps_mask) > 0:
        aps_data = parsed_data[aps_mask]
        aps_reset = numpy.bitwise_and(aps_data["d2"], 0b1100) == 0b0000
        aps_signal = numpy.bitwise_and(aps_data["d2"], 0b1100) == 0b0100
        aps_x = Aerdat2.extract_x(aps_data)
        aps_y = Aerdat2.extract_y(aps_data)
        aps_sample = numpy.bitwise_or(
            numpy.left_shift(numpy.bitwise_and(aps_data["d2"].astype("<u2"), 0b11), 8),
            aps_data["d3"].astype("<u2"),
        )
        t_frames = []
        current_frame = numpy.zeros((240, 180), dtype="<u2")
        first_reset = -1
        last_reset = -1
        first_signal = -1
        last_signal = -1
        for index in range(0, len(aps_data)):
            x = aps_x[index]
            y = aps_y[index]
            if x < 240 and y < 180:
                if aps_reset[index]:
                    if first_reset == -1:
                        first_reset = aps_data["t"][index]
                    if x == 0 and y == 0:
                        if (
                            first_reset != -1
                            and last_reset != -1
                            and first_signal != -1
                            and last_signal != -1
                        ):
                            t_frames.append(
                                (
                                    numpy.copy(current_frame),
                                    first_reset,
                                    last_reset,
                                    first_signal,
                                    last_signal,
                                )
                            )
                        current_frame.fill(0)
                        first_reset = aps_data["t"][index]
                        last_reset = -1
                        first_signal = -1
                        last_signal = -1
                    else:
                        last_reset = aps_data["t"][index]
                    current_frame[x, y] = aps_sample[index]
                elif aps_signal[index]:
                    if first_signal == -1:
                        first_signal = aps_data["t"][index]
                    last_signal = aps_data["t"][index]
                    luma = current_frame[x, y]
                    sample = aps_sample[index]
                    if sample > luma:
                        luma = 0
                    else:
                        luma -= sample
                    current_frame[x, y] = luma
        if (
            first_reset != -1
            and last_reset != -1
            and first_signal != -1
            and last_signal != -1
        ):
            t_frames.append(
                (current_frame, first_reset, last_reset, first_signal, last_signal)
            )
        if len(t_frames) == 0:
            aedat2.aps_corrupted = True
        else:
            aedat2.aps = numpy.zeros(len(t_frames), dtype=undr.raw.aps_dtype(240, 180))
            aedat2.aps["width"].fill(240)
            aedat2.aps["height"].fill(180)
            for index in range(0, len(t_frames)):
                aedat2.aps["t"][index] = t_frames[index][3]
                aedat2.aps["begin_t"][index] = t_frames[index][1]
                aedat2.aps["end_t"][index] = t_frames[index][3]
                aedat2.aps["exposure_begin_t"][index] = t_frames[index][2]
                aedat2.aps["exposure_end_t"][index] = t_frames[index][4]
                aedat2.aps["pixels"][index] = t_frames[index][0]
    if numpy.count_nonzero(imu_mask) > 0:
        imu_data = parsed_data[imu_mask]
        timestamps, indices, counts = numpy.unique(
            imu_data["t"], return_index=True, return_counts=True
        )
        unique_mask = counts == 7
        if numpy.count_nonzero(unique_mask) > 0:
            samples = numpy.bitwise_or(
                numpy.left_shift(
                    numpy.bitwise_and(imu_data["d0"].astype("<u2"), 0b1111), 12
                ),
                numpy.bitwise_or(
                    numpy.left_shift(imu_data["d1"].astype("<u2"), 4),
                    numpy.right_shift(imu_data["d2"].astype("<u2"), 4),
                ),
            )
            types = numpy.right_shift(numpy.bitwise_and(imu_data["d0"], 0b1110000), 4)
            boundary_indices = indices[unique_mask]
            valid = True
            for offset in range(0, 7):
                if not numpy.all(types[boundary_indices + offset] == offset):
                    aedat2.imu_corrupted = True
                    valid = False
                    break
            if valid:
                aedat2.imu = numpy.zeros(
                    numpy.count_nonzero(unique_mask), dtype=undr.raw.IMU_DTYPE
                )
                aedat2.imu["t"] = timestamps[unique_mask]
                aedat2.imu["accelerometer_x"] = samples[boundary_indices]
                aedat2.imu["accelerometer_y"] = samples[boundary_indices + 1]
                aedat2.imu["accelerometer_z"] = samples[boundary_indices + 2]
                aedat2.imu["temperature"] = samples[boundary_indices + 3]
                aedat2.imu["gyroscope_x"] = samples[boundary_indices + 4]
                aedat2.imu["gyroscope_y"] = samples[boundary_indices + 5]
                aedat2.imu["gyroscope_z"] = samples[boundary_indices + 6]
    return aedat2
