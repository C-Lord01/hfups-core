"""Reindex flood_v2 label files to match the new 15-class sequential vocabulary.

Changes:
  12 (carpark)         -> 11
  13 (ocean)           -> 12
  14 (waves)           -> 13
  15 (debris_floating) -> 14
  11 (utility_pole)    -> DELETE line
  16 (rescue_boat)     -> DELETE line
  0-10                 -> unchanged
"""

from __future__ import annotations

from pathlib import Path

FLOOD_V2_DIR = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\flood_v2"
)

OLD_TO_NEW = {12: 11, 13: 12, 14: 13, 15: 14}
DELETE_CLASSES = {11, 16}  # utility_pole, rescue_boat


def main() -> None:
    total_files_updated = 0
    total_lines_remapped = 0
    total_lines_deleted = 0
    total_files_scanned = 0

    for split in ("train", "valid", "test"):
        lbl_dir = FLOOD_V2_DIR / split / "labels"
        if not lbl_dir.exists():
            print(f"  WARNING: {lbl_dir} not found, skipping.")
            continue

        for lbl_file in sorted(lbl_dir.iterdir()):
            if lbl_file.suffix != ".txt":
                continue

            total_files_scanned += 1
            original = lbl_file.read_text(encoding="utf-8")
            new_lines: list[str] = []
            file_remapped = 0
            file_deleted = 0
            changed = False

            for line in original.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                cls_id = int(parts[0])

                if cls_id in DELETE_CLASSES:
                    file_deleted += 1
                    changed = True
                    continue

                if cls_id in OLD_TO_NEW:
                    new_cls = OLD_TO_NEW[cls_id]
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
                    file_remapped += 1
                    changed = True
                else:
                    new_lines.append(stripped)

            if changed:
                lbl_file.write_text(
                    "\n".join(new_lines) + ("\n" if new_lines else ""),
                    encoding="utf-8",
                )
                total_files_updated += 1
                total_lines_remapped += file_remapped
                total_lines_deleted += file_deleted

    print(f"Scanned:       {total_files_scanned:,} label files")
    print(f"Files updated: {total_files_updated:,}")
    print(f"Lines remapped (old idx -> new idx): {total_lines_remapped:,}")
    print(f"Lines deleted  (utility_pole / rescue_boat): {total_lines_deleted:,}")


if __name__ == "__main__":
    main()
