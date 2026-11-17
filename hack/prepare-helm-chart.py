"""Prepare the bundled Higress workload templates and Collector config."""

import io
import sys
import tarfile
import tempfile
from pathlib import Path


COLLECTOR_CONFIG = "higress-core/gpustack-collector.yaml"


def prepare_higress(archive: Path, config: bytes) -> None:
    """Expose gateway templates to the parent and bundle the shared log parser."""
    templates = {
        f"higress-core/templates/{kind.lower()}.yaml": (
            kind,
            f'{{{{- if eq .Values.gateway.kind "{kind}" -}}}}'.encode(),
        )
        for kind in ("Deployment", "DaemonSet")
    }
    templates["higress-core/templates/configmap.yaml"] = (
        "ConfigMap",
        b'{{- $originalMesh := include "mesh" . | fromYaml }}',
    )
    with tempfile.TemporaryDirectory(dir=archive.parent) as temp_dir:
        prepared = Path(temp_dir) / archive.name
        found = set()
        with (
            tarfile.open(archive, "r:gz") as source,
            tarfile.open(prepared, "w:gz") as target,
        ):
            for member in source:
                if member.name == COLLECTOR_CONFIG:
                    continue
                content = source.extractfile(member) if member.isfile() else None
                if member.name in templates:
                    if content is None:
                        raise ValueError(f"Not a file: {member.name}")
                    found.add(member.name)
                    kind, anchor = templates[member.name]
                    opening = f'{{{{- define "gpustack.higress.{kind}" -}}}}\n'.encode()
                    body = content.read()
                    before, match, after = body.partition(anchor)
                    if not match:
                        raise ValueError(f"Unexpected Higress template: {member.name}")
                    if not before.endswith(opening):
                        body = (
                            before
                            + opening
                            + match
                            + after.rstrip()
                            + b"\n{{- end }}\n"
                        )
                    member.size = len(body)
                    content = io.BytesIO(body)
                target.addfile(member, content)
            entry = tarfile.TarInfo(COLLECTOR_CONFIG)
            entry.size = len(config)
            target.addfile(entry, io.BytesIO(config))
        if found != templates.keys():
            raise ValueError("The Higress chart is missing gateway templates")
        prepared.replace(archive)


def main(chart: Path) -> None:
    archives = list((chart / "charts").glob("higress-core-*.tgz"))
    if len(archives) != 1:
        raise ValueError("Expected one Higress archive; run helm dependency update")
    # The Helm template selects the gateway pipeline from the same config as Docker.
    config = (
        Path(__file__).resolve().parents[1]
        / "pack/rootfs/etc/gpustack/logging/collector.yaml"
    )
    prepare_higress(archives[0], config.read_bytes())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
