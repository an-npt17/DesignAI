from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JsonObject = dict[str, object]
QuaternionFormat = Literal["object", "array"]

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "demo_inventory"
DEFAULT_INPUT_PATTERN = "*.orientation.json"
DEFAULT_OUTPUT_KEY = "rotation_quaternion_offset"


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class ConversionResult:
    path: Path
    changed: bool
    converted_count: int


def yaw_degrees_to_quaternion(rotation_deg_offset: float) -> Quaternion:
    half_angle_rad = math.radians(rotation_deg_offset) / 2.0
    return Quaternion(
        x=0.0,
        y=math.sin(half_angle_rad),
        z=0.0,
        w=math.cos(half_angle_rad),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert rotation_deg_offset values in orientation JSON files to "
            "yaw quaternions."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_INPUT],
        help=(
            "Files or directories to process. Directories are searched for "
            f"{DEFAULT_INPUT_PATTERN}."
        ),
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_INPUT_PATTERN,
        help="Glob used when an input path is a directory.",
    )
    parser.add_argument(
        "--output-key",
        default=DEFAULT_OUTPUT_KEY,
        help="Sibling key written next to each rotation_deg_offset.",
    )
    parser.add_argument(
        "--format",
        choices=("object", "array"),
        default="object",
        help='Use {"x","y","z","w"} or [x,y,z,w] quaternion JSON.',
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=10,
        help="Decimal places used for quaternion components.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite files in place. Without this flag the script only previews.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(iter_orientation_json_paths(args.paths, pattern=args.pattern))
    if not paths:
        raise SystemExit("No orientation JSON files matched the provided path(s).")

    results = [
        convert_file(
            path=path,
            output_key=str(args.output_key),
            quaternion_format=parse_quaternion_format(args.format),
            precision=int(args.precision),
            write=bool(args.write),
        )
        for path in paths
    ]

    action = "updated" if args.write else "would update"
    for result in results:
        if result.converted_count == 0:
            print(f"skipped {result.path}: no rotation_deg_offset values")
            continue
        status = action if result.changed else "already current"
        print(f"{status} {result.path}: {result.converted_count} quaternion(s)")

    return 0


def iter_orientation_json_paths(
    paths: Sequence[Path],
    *,
    pattern: str,
) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        candidates = sorted(path.glob(pattern)) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            yield resolved


def convert_file(
    *,
    path: Path,
    output_key: str,
    quaternion_format: QuaternionFormat,
    precision: int,
    write: bool,
) -> ConversionResult:
    payload = read_json_object(path)
    converted_count = add_quaternion_offsets(
        payload,
        output_key=output_key,
        quaternion_format=quaternion_format,
        precision=precision,
    )
    next_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    current_text = path.read_text(encoding="utf-8")
    changed = current_text != next_text
    if write and changed:
        path.write_text(next_text, encoding="utf-8")
    return ConversionResult(
        path=path,
        changed=changed,
        converted_count=converted_count,
    )


def parse_quaternion_format(value: object) -> QuaternionFormat:
    if value == "object" or value == "array":
        return value
    raise ValueError(f"Unsupported quaternion format: {value}")


def read_json_object(path: Path) -> JsonObject:
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(raw_payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return {str(key): value for key, value in raw_payload.items()}


def add_quaternion_offsets(
    value: object,
    *,
    output_key: str,
    quaternion_format: QuaternionFormat,
    precision: int,
) -> int:
    if isinstance(value, list):
        return sum(
            add_quaternion_offsets(
                item,
                output_key=output_key,
                quaternion_format=quaternion_format,
                precision=precision,
            )
            for item in value
        )
    if not isinstance(value, dict):
        return 0

    converted_count = 0
    rotation_deg_offset = read_number(value.get("rotation_deg_offset"))
    if rotation_deg_offset is not None:
        value[output_key] = build_quaternion_payload(
            yaw_degrees_to_quaternion(rotation_deg_offset),
            quaternion_format=quaternion_format,
            precision=precision,
        )
        converted_count += 1

    for item in value.values():
        converted_count += add_quaternion_offsets(
            item,
            output_key=output_key,
            quaternion_format=quaternion_format,
            precision=precision,
        )
    return converted_count


def read_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def build_quaternion_payload(
    quaternion: Quaternion,
    *,
    quaternion_format: QuaternionFormat,
    precision: int,
) -> JsonObject | list[float]:
    components = [
        rounded_component(quaternion.x, precision=precision),
        rounded_component(quaternion.y, precision=precision),
        rounded_component(quaternion.z, precision=precision),
        rounded_component(quaternion.w, precision=precision),
    ]
    if quaternion_format == "array":
        return components
    if quaternion_format == "object":
        return {
            "x": components[0],
            "y": components[1],
            "z": components[2],
            "w": components[3],
        }
    raise ValueError(f"Unsupported quaternion format: {quaternion_format}")


def rounded_component(value: float, *, precision: int) -> float:
    rounded = round(value, precision)
    if rounded == 0:
        return 0.0
    return rounded


if __name__ == "__main__":
    raise SystemExit(main())
