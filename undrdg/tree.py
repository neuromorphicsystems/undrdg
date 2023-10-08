"""Utilities to convert a file tree."""

from __future__ import annotations

import collections
import dataclasses
import multiprocessing.pool
import pathlib
import typing

from . import path


class Rule(typing.Protocol):
    def match(self, source_relative_path: pathlib.Path) -> bool:
        ...

    def skip(self) -> bool:
        ...

    def rename(self, name: str) -> str:
        ...


class Rename:
    def __init__(
        self,
        match_relative_path: typing.Union[str, pathlib.Path],
        new_name: str,
    ):
        self.match_relative_path = pathlib.Path(match_relative_path)
        self.new_name = new_name

    def match(self, source_relative_path: pathlib.Path) -> bool:
        return source_relative_path == self.match_relative_path

    def skip(self) -> bool:
        return False

    def rename(self, name: str) -> str:
        return self.new_name


class RenameExtension:
    def __init__(self, match_extension: str, new_extension: str):
        self.match_extension = match_extension
        self.new_extension = new_extension

    def match(self, source_relative_path: pathlib.Path) -> bool:
        return source_relative_path.suffix == self.match_extension

    def skip(self) -> bool:
        return False

    def rename(self, name: str) -> str:
        name_as_path = pathlib.Path(name)
        return f"{name_as_path.stem}{self.new_extension}"


class SkipName:
    def __init__(
        self,
        match_name: str,
    ):
        self.match_name = match_name

    def match(self, source_relative_path: pathlib.Path) -> bool:
        return source_relative_path.name == self.match_name

    def skip(self) -> bool:
        return True

    def rename(self, name: str) -> str:
        return name


@dataclasses.dataclass
class Task:
    source: pathlib.Path
    target_directory: path.Directory
    target_name: str
    index: int
    total: int


def tasks(
    source: pathlib.Path,
    target: path.Directory,
    rules: list[Rule],
    source_root: pathlib.Path,
    target_root: pathlib.Path,
    debouncer: typing.Optional[path.Debouncer],
) -> typing.Iterator[Task]:
    node_sources = list(source.iterdir())
    node_sources.sort(
        key=lambda node_source: f"1{node_source.name}"
        if node_source.is_dir()
        else f"2{node_source.name}"
    )
    directories_nodes: list[tuple[pathlib.Path, str]] = []
    for node_source in node_sources:
        source_relative_path = node_source.relative_to(source_root)
        target_name = node_source.name
        skip = False
        for rule in rules:
            if rule.match(source_relative_path):
                if rule.skip():
                    skip = True
                    break
                target_name = rule.rename(target_name)
        if skip:
            continue
        if node_source.is_dir():
            directories_nodes.append((node_source, target_name))
        else:
            yield Task(
                source=node_source,
                target_directory=target,
                target_name=target_name,
                index=0,
                total=0,
            )
    for node_source, target_name in directories_nodes:
        yield from tasks(
            source=node_source,
            target=target.create_subdirectory(target_name, debouncer=debouncer),
            rules=rules,
            source_root=source_root,
            target_root=target_root,
            debouncer=debouncer,
        )


def copy_tree(
    source: pathlib.Path,
    target: pathlib.Path,
    rules: list[Rule],
    handle_task: typing.Callable[[Task], None],
    thread_pool_processes: typing.Optional[int] = None,
    pool_imap_chunksize: int = 1,
):
    with path.Debouncer() as debouncer:
        target_directory = path.Directory(path=target, debouncer=debouncer)
        tasks_list = list(
            tasks(
                source=source,
                target=target_directory,
                rules=rules,
                source_root=source,
                target_root=target,
                debouncer=debouncer,
            )
        )
        for index, task in enumerate(tasks_list):
            task.index = index
            task.total = len(tasks_list)
        with multiprocessing.pool.ThreadPool(processes=thread_pool_processes) as pool:
            collections.deque(
                pool.map(
                    handle_task,
                    tasks_list,
                    chunksize=pool_imap_chunksize,
                ),
                maxlen=0,
            )
