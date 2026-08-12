"""Declarative JSON/YAML source catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .base import SourceAdapter
from .git_repository import GitRepositorySource
from .local_directory import LocalDirectorySource
from .official_web import Fetcher, OfficialWebSource


@dataclass(slots=True)
class CatalogEntry:
    source_id: str
    source_type: str
    options: dict[str, Any] = field(default_factory=dict)


class SourceCatalog:
    def __init__(self, entries: Iterable[CatalogEntry], *, base_dir: str | Path = ".") -> None:
        self.entries = list(entries)
        self.base_dir = Path(base_dir).resolve()
        identifiers = [entry.source_id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("source catalog contains duplicate IDs")

    @classmethod
    def load(cls, path: str | Path) -> "SourceCatalog":
        catalog_path = Path(path).resolve()
        raw = catalog_path.read_text(encoding="utf-8")
        suffix = catalog_path.suffix.lower()
        if suffix == ".json":
            data = json.loads(raw)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - project dependencies provide PyYAML
                raise RuntimeError("PyYAML is required to read YAML source catalogs") from exc
            data = yaml.safe_load(raw)
        else:
            raise ValueError(f"unsupported source catalog format: {catalog_path.suffix}")
        return cls.from_mapping(data, base_dir=catalog_path.parent)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | list[Mapping[str, Any]],
        *,
        base_dir: str | Path = ".",
    ) -> "SourceCatalog":
        sources = value.get("sources") if isinstance(value, Mapping) else value
        if not isinstance(sources, list):
            raise ValueError("source catalog must contain a 'sources' list")
        entries = []
        for item in sources:
            if not isinstance(item, Mapping):
                raise ValueError("each source catalog entry must be a mapping")
            source_id = str(item.get("id") or item.get("source_id") or "").strip()
            source_type = str(item.get("type") or item.get("source_type") or "").strip()
            if not source_id or not source_type:
                raise ValueError("each source requires id and type")
            options = {key: _expand_values(val) for key, val in item.items() if key not in {"id", "source_id", "type", "source_type"}}
            entries.append(CatalogEntry(source_id=source_id, source_type=source_type, options=options))
        return cls(entries, base_dir=base_dir)

    def create_sources(self, *, web_fetcher: Fetcher | None = None) -> list[SourceAdapter]:
        adapters: list[SourceAdapter] = []
        for entry in self.entries:
            options = dict(entry.options)
            source_type = entry.source_type.lower().replace("-", "_")
            if source_type in {"git", "git_repository"}:
                raw_path = options.pop("path", options.pop("repository_path", None))
                if raw_path is None:
                    raise ValueError(f"git source '{entry.source_id}' requires path")
                repository_path = Path(str(raw_path)).expanduser()
                if not repository_path.is_absolute():
                    repository_path = self.base_dir / repository_path
                if "python_symbol_cards" in options:
                    options["include_python_symbols"] = options.pop("python_symbol_cards")
                adapters.append(
                    GitRepositorySource(
                        entry.source_id,
                        repository_path,
                        **options,
                    )
                )
            elif source_type in {"directory", "local", "local_directory"}:
                raw_path = options.pop(
                    "path",
                    options.pop("directory_path", options.pop("root", None)),
                )
                if raw_path is None:
                    raise ValueError(
                        f"local directory source '{entry.source_id}' requires path"
                    )
                directory_path = Path(str(raw_path)).expanduser()
                if not directory_path.is_absolute():
                    directory_path = self.base_dir / directory_path
                adapters.append(
                    LocalDirectorySource(
                        entry.source_id,
                        directory_path,
                        **options,
                    )
                )
            elif source_type in {"web", "official_web"}:
                urls = options.pop("urls", None)
                allowed_domains = options.pop("allowed_domains", None)
                if not urls or not allowed_domains:
                    raise ValueError(
                        f"official web source '{entry.source_id}' requires urls and allowed_domains"
                    )
                adapters.append(
                    OfficialWebSource(
                        entry.source_id,
                        urls,
                        allowed_domains=allowed_domains,
                        fetcher=web_fetcher,
                        **options,
                    )
                )
            else:
                raise ValueError(f"unsupported source type: {entry.source_type}")
        return adapters


def _expand_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_values(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _expand_values(item) for key, item in value.items()}
    return value
