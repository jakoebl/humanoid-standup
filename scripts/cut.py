#!/usr/bin/env python3
"""Cut an inclusive, zero-based frame range from a motion CSV."""

import argparse
import csv
from pathlib import Path


def cut_clip(source: Path, start: int, end: int) -> Path:
    """Save selected rows beside the source, with Frame numbering reset to zero."""
    if start < 0 or end < start:
        raise ValueError("Use 0 <= start <= end (inclusive frame indices).")
    if not source.exists() and source.suffix != ".csv":
        source = Path(f"{source}.csv")
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or "Frame" not in header:
            raise ValueError("Expected a motion CSV with a header containing 'Frame'.")
        rows = [row for row in reader if row]
    if end >= len(rows):
        raise ValueError(f"End frame {end} is out of range; this clip has {len(rows)} frames.")
    selected = rows[start:end + 1]
    frame_column = header.index("Frame")
    for index, row in enumerate(selected):
        if len(row) != len(header):
            raise ValueError(f"Frame {start + index} has a different column count from the header.")
        row[frame_column] = str(index)
    output = source.with_name(f"standup_{start}_{end}.csv")
    # Exclusive creation also protects the original when its name matches the output.
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(selected)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_clip", type=Path, help="Source CSV path (.csv may be omitted).")
    parser.add_argument("start", type=int, help="First frame index, starting at zero.")
    parser.add_argument("end", type=int, help="Last frame index, inclusive.")
    args = parser.parse_args()
    try:
        output = cut_clip(args.original_clip, args.start, args.end)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {args.end - args.start + 1} frames to {output}")


if __name__ == "__main__":
    main()
