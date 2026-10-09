"""Convert raw G1 clips to mjlab's headerless XYZ/XYZW/29-joint CSV format.

Legacy named CSVs use centimeters and XYZ Euler/joint angles in degrees.
Headerless quaternion CSVs already use meters and radians. Sampling and frame
counts are preserved: supply the original FPS separately to mjlab.csv_to_npz.
This script does not generate the MuJoCo body-kinematics NPZ.
"""

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path


JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")] + [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
]
LEGACY_COLUMNS = (
    [f"root_translate{axis}" for axis in "XYZ"]
    + [f"root_rotate{axis}" for axis in "XYZ"]
    + [f"{name}_dof" for name in JOINT_NAMES]
)


def convert_rows(source):
    with source.open(newline="") as stream:
        rows = [row for row in csv.reader(stream) if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("Empty clip.")
    legacy = "root_translateX" in rows[0]
    if legacy:
        header = rows.pop(0)
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names.")
        missing = set(LEGACY_COLUMNS) - set(header)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        indices = [header.index(name) for name in LEGACY_COLUMNS]
    if len(rows) < 3:
        raise ValueError("At least three frames are required for mjlab's velocity calculation.")

    output = []
    previous_quat = None
    for index, row in enumerate(rows):
        expected = len(header) if legacy else 36
        if len(row) != expected:
            raise ValueError(f"Frame {index}: expected {expected} columns, got {len(row)}.")
        values = [float(row[i]) for i in indices] if legacy else [float(cell) for cell in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Frame {index}: nonfinite motion value.")
        if legacy:
            position = [value / 100 for value in values[:3]]
            roll, pitch, yaw = [math.radians(value) for value in values[3:6]]
            cr, cp, cy = [math.cos(angle / 2) for angle in (roll, pitch, yaw)]
            sr, sp, sy = [math.sin(angle / 2) for angle in (roll, pitch, yaw)]
            quat = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                    cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
            joints = [math.radians(value) for value in values[6:]]
        else:
            position, quat, joints = values[:3], values[3:7], values[7:]
        norm = math.hypot(*quat)
        if norm < 1e-12:
            raise ValueError(f"Frame {index}: zero-length quaternion.")
        quat = [value / norm for value in quat]
        if previous_quat is not None and sum(a*b for a, b in zip(previous_quat, quat)) < 0:
            quat = [-value for value in quat]
        previous_quat = quat
        output.append(position + quat + joints)
    return output, "legacy Euler" if legacy else "quaternion"


def write_clip(destination, rows):
    """Replace the destination only after its complete output is written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", dir=destination.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            csv.writer(stream).writerows(rows)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/unprocessed")
    parser.add_argument("--output-dir", type=Path, default=root / "data/mjlab")
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Input and output directories must differ.")
    sources = sorted(args.input_dir.glob("*.csv"))
    if not sources:
        parser.error(f"No CSV clips found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for source in sources:
        try:
            destination = args.output_dir / source.name
            if destination.resolve() == source.resolve():
                raise ValueError("Output must not overwrite the source clip.")
            rows, kind = convert_rows(source)
            write_clip(destination, rows)
            print(f"{source.name}: {len(rows)} frames ({kind}) -> {destination}")
        except (OSError, ValueError, csv.Error) as error:
            failures += 1
            print(f"FAILED {source.name}: {error}")
    print(f"Converted {len(sources) - failures}/{len(sources)} clips; sampling rates preserved.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
