"""File system wrappers."""

from __future__ import annotations

import bisect
import copy
import itertools
import json
import operator
import pathlib
import threading
import time
import types
import typing

import brotli
import numpy
import undr

DEFAULT_INDEX_DATA = {
    "directories": [],
    "files": [],
    "other_files": [],
    "version": {"major": 1, "minor": 0, "patch": 0},
}


class Type(typing.Protocol):
    """A file type supported by UNDR."""

    def dtype(self) -> numpy.dtype:
        ...

    def extension(self) -> str:
        ...

    def properties(self) -> dict[str, typing.Any]:
        ...


class DvsType:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def dtype(self) -> numpy.dtype:
        return undr.raw.DVS_DTYPE

    def extension(self) -> str:
        return "dvs"

    def properties(self) -> dict[str, typing.Any]:
        return {
            "type": "dvs",
            "width": self.width,
            "height": self.height,
        }


class ApsType:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def dtype(self) -> numpy.dtype:
        return undr.raw.aps_dtype(width=self.width, height=self.height)

    def extension(self) -> str:
        return "aps"

    def properties(self) -> dict[str, typing.Any]:
        return {
            "type": "aps",
            "width": self.width,
            "height": self.height,
        }


class ImuType:
    def dtype(self) -> numpy.dtype:
        return undr.raw.IMU_DTYPE

    def extension(self) -> str:
        return "imu"

    def properties(self) -> dict[str, typing.Any]:
        return {
            "type": "imu",
        }


class Directory:
    """A local directory.

    Args:
        path (typing.Union[pathlib.Path, str]): The path of the directory in the local file system.
        debouncer (typing.Optional[Debouncer], optional): Delegate that saves the index to the disk to optimize frequent writes. Defaults to None.
    """

    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        debouncer: typing.Optional["Debouncer"] = None,
    ):
        if isinstance(path, str):
            self.path = pathlib.Path(path)
        else:
            self.path = path
        self.path.mkdir(exist_ok=True, parents=False)
        self.index_path = self.path / "-index.json"
        if not self.index_path.exists():
            default_index_content = (
                f"{json.dumps(DEFAULT_INDEX_DATA, sort_keys=True, indent=4)}\n".encode()
            )
            with open(self.index_path, "wb") as index_data_file:
                index_data_file.write(default_index_content)
        self.index_data = undr.json_index.load(self.index_path)
        names = set()
        for directory in self.index_data["directories"]:
            if directory in names:
                raise Exception(
                    f'duplicate file or directory "{directory}" in {self.index_path}'
                )
            names.add(directory)
        self.index_data["directories"].sort()
        for file in itertools.chain(
            self.index_data["files"], self.index_data["other_files"]
        ):
            if file["name"] in names:
                raise Exception(
                    f"duplicate file or directory \"{file['name']}\" in {self.index_path}"
                )
            names.add(file["name"])
        self.index_data["files"].sort(key=operator.itemgetter("name"))
        self.index_data["other_files"].sort(key=operator.itemgetter("name"))
        self.added_names = set()
        self.index_update_lock = threading.Lock()
        self.index_write_lock = threading.Lock()
        self.debouncer = debouncer
        if self.debouncer is not None:
            self.debouncer_id = self.debouncer.register(self)
        else:
            self.debouncer_id = 0

    def create_subdirectory(
        self, name: str, debouncer: typing.Optional["Debouncer"] = None
    ) -> "Directory":
        with self.index_update_lock:
            if name in self.added_names:
                raise Exception(
                    f'attempted to add the file or directory "{name}" twice in {self.index_path}'
                )
            self.added_names.add(name)
            insertion_point = bisect.bisect_left(self.index_data["directories"], name)
            if insertion_point == len(self.index_data["directories"]):
                self.index_data["directories"].append(name)
            elif name != self.index_data["directories"][insertion_point]:
                self.index_data["directories"].insert(insertion_point, name)
            if self.debouncer is None:
                self.save_index_data(index_update_lock_already_acquired=True)
            else:
                self.debouncer.set(self.debouncer_id)
        return Directory(path=self.path / name, debouncer=debouncer)

    def create_file(
        self, type: Type, name: str, metadata: dict[str, typing.Any]
    ) -> "File":
        """Adds a file to this directory.

        The file is a context manager compatible with the 'with' statement.
        Upon exiting the context manaager, the file calls :py:meth:`update_index`
        to register itself in the index of this directory.
        If a file with the same name exists, it is silently replaced.

        This function is thread-safe.

        Args:
            type (Type): A type object compatible with UNDR (:py:class:`DvsType`, :py:class:`ApsType`, or :py:class:`ImuType`).
            name (str): The file name without extension (the extension is automatically set from the type).
            metadata (dict[str, typing.Any]): File properties to add to the index.

        Returns:
            File: A writable file.
        """
        return File(directory=self, type=type, name=name, metadata=metadata)

    def create_other_file(
        self, name: str, metadata: dict[str, typing.Any]
    ) -> "OtherFile":
        """Adds a file to this directory.

        The file is a context manager compatible with the 'with' statement.
        Upon exiting the context manaager, the file calls :py:meth:`update_index`
        to register itself in the index of this directory.
        If a file with the same name exists, it is silently replaced.

        This function is thread-safe.

        Args:
            name (str): The file name including the extension.

        Returns:
            OtherFile: A writable file.
        """
        return OtherFile(directory=self, name=name, metadata=metadata)

    def update_index(self, file: "BaseFile"):
        """Adds a file to the index.

        This function is called automatically by files used as context managers ('with').
        It is thread-safe.

        Args:
            file (str): The file to add to the index.
        """
        with self.index_update_lock:
            if isinstance(file, File):
                key = "files"
            elif isinstance(file, OtherFile):
                key = "other_files"
            else:
                raise Exception(f"unsupported base type for file {file._path}")
            if file._index_entry["name"] in self.added_names:
                raise Exception(
                    f"attempted to add the file or directory \"{file._index_entry['name']}\" twice in {self.index_path}"
                )
            self.added_names.add(file._index_entry["name"])
            assert file._index_entry["hash"] is not None
            insertion_point = bisect.bisect_left(
                self.index_data[key],
                file._index_entry["name"],
                key=operator.itemgetter("name"),
            )
            if insertion_point == len(self.index_data[key]):
                self.index_data[key].append(file._index_entry)
            elif (
                self.index_data[key][insertion_point]["name"]
                == file._index_entry["name"]
            ):
                self.index_data[key][insertion_point] = file._index_entry
            else:
                self.index_data[key].insert(insertion_point, file._index_entry)
            if self.debouncer is None:
                self.save_index_data(index_update_lock_already_acquired=True)
            else:
                self.debouncer.set(self.debouncer_id)

    def save_index_data(self, index_update_lock_already_acquired: bool):
        """Write the index to the disk.

        This function is called automatically by files used as context managers ('with').
        It is thread-safe.

        Args:
            index_update_lock_already_acquired bool): Whether the caller has already acquired ``index_update_lock``.
        """
        if index_update_lock_already_acquired:
            index_data = copy.deepcopy(self.index_data)
            self.index_write_lock.acquire()
        else:
            with self.index_update_lock:
                index_data = copy.deepcopy(self.index_data)
                self.index_write_lock.acquire()
        try:
            index_content = (
                f"{json.dumps(index_data, sort_keys=True, indent=4)}\n".encode()
            )
            with open(self.index_path, "wb") as index_data_file:
                index_data_file.write(index_content)
        finally:
            self.index_write_lock.release()


class Debouncer:
    def __init__(self, debounce_period: float = 1.0, sleep_duration: float = 0.1):
        self.debounce_period = debounce_period
        self.sleep_duration = sleep_duration
        self.lock = threading.Lock()
        self.directories: list[Directory] = []
        self.flags = []
        self.running = True
        self.worker = threading.Thread(target=self.target)
        self.worker.daemon = True
        self.worker.start()

    def register(self, directory: Directory) -> int:
        with self.lock:
            self.directories.append(directory)
            self.flags.append(False)
            return len(self.directories) - 1

    def set(self, id: int):
        with self.lock:
            self.flags[id] = True

    def target(self):
        """Worker thread implementation."""
        next_save = time.monotonic()
        while True:
            now = time.monotonic()
            running = copy.copy(self.running)
            if now >= next_save or not running:
                directories_to_update: list[Directory] = []
                with self.lock:
                    for index, flag in enumerate(self.flags):
                        if flag:
                            directories_to_update.append(self.directories[index])
                            self.flags[index] = False
                for directory in directories_to_update:
                    directory.save_index_data(index_update_lock_already_acquired=False)
                next_save += self.debounce_period
                if not running:
                    break
            time.sleep(self.sleep_duration)

    def close(self):
        """Terminates the debouncer and perform all pending updates.

        This function is called automatically if Debouncer is used as a context manager.
        """
        self.running = False
        self.worker.join()

    def __enter__(self) -> "Debouncer":
        """Enables the use of the "with" statement.

        Returns:
            Debouncer: A debouncer context that calls :py:meth:`close` on exit.
        """
        return self

    def __exit__(
        self,
        type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ):
        """Enables the use of the "with" statement.

        Args:
            type (typing.Optional[typing.Type[BaseException]]): None if the context exits without an exception, and the raised exception's class otherwise.
            value (typing.Optional[BaseException]): None if the context exits without an exception, and the raised exception otherwise.
            traceback (typing.Optional[types.TracebackType]): None if the context exits without an exception, and the raised exception's traceback otherwise.
        """
        self.close()


class BaseFile:
    """BaseFile implements the logic common to :py:class:`File` and :py:class:`OtherFile`.

    Args:
        directory (Directory): directory that contains the file.
    """

    def __init__(
        self, directory: Directory, name: str, metadata: dict[str, typing.Any]
    ):
        self._directory = directory
        self._path = self._directory.path / f"{name}.br"
        self._write_path = self._directory.path / f"{name}.br.write"
        self._handle = open(self._write_path, "wb")
        self._compressor = brotli.Compressor(
            mode=brotli.MODE_GENERIC, quality=11, lgwin=22, lgblock=0
        )
        self._index_entry = {
            "compressions": [
                {
                    "hash": None,
                    "size": 0,
                    "suffix": ".br",
                    "type": "brotli",
                }
            ],
            "hash": None,
            "metadata": metadata,
            "name": name,
            "size": 0,
        }
        self.uncompressed_hash_object = undr.utilities.new_hash()
        self.compressed_hash_object = undr.utilities.new_hash()

    def close(self):
        """Closes the file, moves it to its final location, and adds it to the index.

        This function is called automatically if BaseFile is used as a context manager.
        """
        if self._handle is None or self._compressor is None:
            raise Exception(f"close called twice for file {self._path}")
        compressed_data = self._compressor.finish()
        self._handle.write(compressed_data)
        self._handle.close()
        self._write_path.replace(self._path)
        self._index_entry["compressions"][0]["size"] += len(compressed_data)
        self.compressed_hash_object.update(compressed_data)
        self._index_entry["hash"] = self.uncompressed_hash_object.hexdigest()
        self._index_entry["compressions"][0][
            "hash"
        ] = self.compressed_hash_object.hexdigest()
        self._directory.update_index(self)
        self._handle = None
        self._compressor = None

    def write_raw(self, data: bytes):
        """Compresses and writes bytes to the file.

        This function may be called any number of times until the file is closed.

        Args:
            data (bytes): Uncompressed data.
        """
        assert self._handle is not None
        assert self._compressor is not None
        compressed_data = self._compressor.process(data)
        self._handle.write(compressed_data)
        self._index_entry["size"] += len(data)
        self._index_entry["compressions"][0]["size"] += len(compressed_data)
        self.uncompressed_hash_object.update(data)
        self.compressed_hash_object.update(compressed_data)


class File(BaseFile):
    def __init__(
        self,
        directory: Directory,
        type: Type,
        name: str,
        metadata: dict[str, typing.Any],
    ):
        super().__init__(
            directory=directory, name=f"{name}.{type.extension()}", metadata=metadata
        )
        self._index_entry["properties"] = type.properties()
        self.dtype = type.dtype()

    def write(self, data: numpy.ndarray):
        """Writes a numpy array to the file.

        This function may be called any number of times until the file is closed.

        Args:
            data (numpy.ndarray): structured numpy array whose dtype is identical to ``self.dtype``.
        """
        assert data.dtype == self.dtype
        self.write_raw(data.tobytes())

    def __enter__(self) -> "File":
        """Enables the use of the "with" statement.

        Returns:
            BaseFile: A file context that calls :py:meth:`close` on exit.
        """
        return self

    def __exit__(
        self,
        type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ):
        """Enables the use of the "with" statement.

        Args:
            type (typing.Optional[typing.Type[BaseException]]): None if the context exits without an exception, and the raised exception's class otherwise.
            value (typing.Optional[BaseException]): None if the context exits without an exception, and the raised exception otherwise.
            traceback (typing.Optional[types.TracebackType]): None if the context exits without an exception, and the raised exception's traceback otherwise.
        """
        if type is None:
            self.close()


class OtherFile(BaseFile):
    def __init__(
        self, directory: Directory, name: str, metadata: dict[str, typing.Any]
    ) -> None:
        super().__init__(directory=directory, name=name, metadata=metadata)

    def write(self, data: typing.Union[str, bytes]):
        """Writes a string or bytes to a file.

        This function may be called any number of times until the file is closed.

        Args:
            data (typing.Union[str, bytes]): string or bytes (strings are encoded with utf-8).
        """
        if isinstance(data, str):
            self.write_raw(data.encode())
        else:
            self.write_raw(data)

    def __enter__(self) -> "OtherFile":
        """Enables the use of the "with" statement.

        Returns:
            BaseFile: A file context that calls :py:meth:`close` on exit.
        """
        return self

    def __exit__(
        self,
        type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ):
        """Enables the use of the "with" statement.

        Args:
            type (typing.Optional[typing.Type[BaseException]]): None if the context exits without an exception, and the raised exception's class otherwise.
            value (typing.Optional[BaseException]): None if the context exits without an exception, and the raised exception otherwise.
            traceback (typing.Optional[types.TracebackType]): None if the context exits without an exception, and the raised exception's traceback otherwise.
        """
        if type is None:
            self.close()
