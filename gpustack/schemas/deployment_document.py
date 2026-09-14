"""Deployment documents: the YAML a set of model deployments is exported to
and imported from.

An entry is a ``ModelCreate`` body minus the server-derived and
environment-bound fields, so a document exported here imports unchanged into
the same cluster. Only the schema and the dump live here; the routes do the
visibility checks and the transaction.
"""

from datetime import datetime, timezone
from types import UnionType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import yaml
from pydantic import BaseModel, ValidationError

from gpustack import __version__
from .models import Model, ModelCreate, ModelPublic
from .source import unknown_keys

# Fields the server derives or that bind a row to one environment. Dropped on
# export and re-derived on import, so a document never carries them.
SERVER_MANAGED_FIELDS = frozenset(
    {
        "id",
        "created_at",
        "updated_at",
        "deleted_at",
        "ready_replicas",
        "meta",
        "cluster_id",
        "owner_principal_id",
        "access_policy",
    }
)

# LoraListEntry fields populated only once an adapter is mounted on an instance.
LORA_RUNTIME_FIELDS = frozenset({"path", "model_file_id"})

# ``name`` first so a reader can tell entries apart at a glance; the rest keep
# the schema's declaration order (ModelSource fields, then the deployment ones).
ENTRY_FIELDS: List[str] = ["name"] + [
    field
    for field in ModelCreate.model_fields
    if field != "name" and field not in SERVER_MANAGED_FIELDS
]


class DeploymentExportRequest(BaseModel):
    ids: Optional[List[int]] = None
    """Deployments to export; omitted means every deployment the caller can see."""
    cluster_id: Optional[int] = None
    """Narrow the export to one cluster."""


def _bare_lora_entry(entry: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    lora_name = entry.get("lora_name") or ""
    if lora_name.startswith(prefix):
        entry["lora_name"] = lora_name[len(prefix) :]
    return {
        key: value for key, value in entry.items() if key not in LORA_RUNTIME_FIELDS
    }


def deployment_entry(model: Model, enable_model_route: bool) -> Dict[str, Any]:
    """One document entry for ``model``: its ``ModelCreate`` fields in
    declaration order with ``None`` values left out. LoRA names are the bare
    short names clients see; import re-adds the stored ``<base>:`` prefix."""
    data = model.model_dump(mode="json", exclude_none=True)
    data["enable_model_route"] = enable_model_route
    if data.get("lora_list"):
        prefix = f"{model.name}:"
        data["lora_list"] = [
            _bare_lora_entry(entry, prefix) for entry in data["lora_list"]
        ]
    return {field: data[field] for field in ENTRY_FIELDS if field in data}


def dump_deployments(
    models: Iterable[Model],
    route_backed_ids: Set[int],
    exported_at: Optional[datetime] = None,
) -> str:
    """Render ``models`` as a deployment document.

    ``route_backed_ids`` are the model ids that have a model route created
    alongside them — what ``enable_model_route`` re-creates on import. Apart
    from the timestamp in the header comment the output is a pure function of
    the rows, so a re-export diffs cleanly.
    """
    if exported_at is None:
        exported_at = datetime.now(timezone.utc)
    header = (
        f"# Exported from GPUStack v{__version__} "
        f"at {exported_at:%Y-%m-%dT%H:%M:%SZ}\n"
    )
    document = {
        "deployments": [
            deployment_entry(model, model.id in route_backed_ids) for model in models
        ]
    }
    return header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


class DeploymentImportRequest(BaseModel):
    content: str
    """The document text; the client reads the file and uploads it as a string."""
    cluster_id: int
    """Cluster every entry is created in; the document itself carries none."""
    dry_run: bool = False
    """Validate only, create nothing."""


class DeploymentImportResult(BaseModel):
    dry_run: bool
    items: List[Union[ModelPublic, ModelCreate]]
    """Created deployments, or on a dry run the entries as they were read."""


class _DeploymentDocument(BaseModel):
    """The top-level shape; anything but ``deployments`` is unknown."""

    deployments: List[Any]


class LoadedDeployments(NamedTuple):
    entries: List[Tuple[int, ModelCreate]]
    """Entries that parsed, each with its index in the document."""
    errors: List[str]
    """One line per problem, labelled ``deployments[i] (<name>): ...``."""


def _validation_error_line(error: Dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else error["msg"]


def _nested_model(annotation: Any) -> Optional[Type[BaseModel]]:
    """The pydantic model a field holds, seen through Optional / List wrappers."""
    if isinstance(annotation, type):
        return annotation if issubclass(annotation, BaseModel) else None
    if get_origin(annotation) in (Union, UnionType, list):
        for argument in get_args(annotation):
            nested = _nested_model(argument)
            if nested is not None:
                return nested
    return None


def _unknown_paths(
    raw: Dict[str, Any], model: Type[BaseModel], prefix: str = ""
) -> List[str]:
    """Keys ``model`` has no field for, at every depth, as ``a.b[0].c`` paths.

    Pydantic ignores extra keys inside nested models too, so a typo in
    ``gpu_selector`` or a new ``lora_list`` option would otherwise vanish.
    """
    paths = [f"{prefix}{key}" for key in unknown_keys(raw, model)]
    for name, field in model.model_fields.items():
        nested = _nested_model(field.annotation)
        value = raw.get(name)
        if nested is None or value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for position, item in enumerate(items):
            if isinstance(item, dict):
                index = f"[{position}]" if isinstance(value, list) else ""
                paths += _unknown_paths(item, nested, f"{prefix}{name}{index}.")
    return paths


def _lora_runtime_paths(raw: Dict[str, Any]) -> List[str]:
    lora_list = raw.get("lora_list")
    if not isinstance(lora_list, list):
        return []
    return [
        f"lora_list[{position}].{key}"
        for position, item in enumerate(lora_list)
        if isinstance(item, dict)
        for key in sorted(LORA_RUNTIME_FIELDS & {str(key) for key in item})
    ]


def load_deployments(text: str) -> LoadedDeployments:
    """Parse a document strictly into ``ModelCreate`` entries.

    A document that is not valid YAML, or not a mapping holding a
    ``deployments`` list, raises ``ValueError`` at once. Entry problems are
    collected instead, alongside the entries that did parse, so the route can
    run its own checks on those and report everything in one pass. Unknown and
    server-managed fields are errors at every depth, not silently dropped: a
    document from a newer GPUStack must not lose configuration on the way in.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"the document is not valid YAML: {e}")
    if not isinstance(document, dict):
        raise ValueError("the document must be a mapping with a 'deployments' list")
    unknown = sorted(unknown_keys(document, _DeploymentDocument))
    if unknown:
        raise ValueError(
            f"unknown top-level field(s): {', '.join(unknown)}; "
            "only 'deployments' is allowed"
        )
    raw_entries = document.get("deployments")
    if not isinstance(raw_entries, list):
        raise ValueError("'deployments' must be a list")

    entries: List[Tuple[int, ModelCreate]] = []
    errors: List[str] = []
    first_index_by_name: Dict[str, int] = {}
    for index, raw in enumerate(raw_entries):
        name = raw.get("name") if isinstance(raw, dict) else None
        label = f"deployments[{index}] ({name})" if name else f"deployments[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{label}: must be a mapping")
            continue
        managed = sorted(str(key) for key in raw if key in SERVER_MANAGED_FIELDS)
        managed += _lora_runtime_paths(raw)
        if managed:
            errors.append(
                f"{label}: server-managed field(s) are not allowed: "
                f"{', '.join(managed)}"
            )
        unknown = sorted(
            path
            for path in _unknown_paths(raw, ModelCreate)
            if path not in SERVER_MANAGED_FIELDS
        )
        if unknown:
            errors.append(f"{label}: unknown field(s): {', '.join(unknown)}")
        try:
            entry = ModelCreate.model_validate(raw)
        except ValidationError as e:
            errors.extend(
                f"{label}: {_validation_error_line(error)}" for error in e.errors()
            )
            continue
        if entry.name in first_index_by_name:
            errors.append(
                f"{label}: duplicate name, already used by "
                f"deployments[{first_index_by_name[entry.name]}]"
            )
            continue
        first_index_by_name[entry.name] = index
        entries.append((index, entry))
    return LoadedDeployments(entries, errors)
