"""
list_files.py
─────────────
1. Lists .h5 / .pkl / .py / .html files in the project ROOT (no subfolders).
2. Then scans the 'asl_webapp' subfolder (and all its children) for the same types.
Output is printed to console and saved to directory_tree.txt
"""

import os

ROOT       = r"C:\Users\kafle\Documents\VS_AI\Hand_sign_to_voice"
WEBAPP_DIR = os.path.join(ROOT, "asl_webapp")
OUTPUT     = "directory_tree.txt"
EXTENSIONS = {'.h5', '.pkl', '.py', '.html'}

lines = []
count_by_ext = {ext: 0 for ext in EXTENSIONS}


def fmt_size(path):
    size_kb = os.path.getsize(path) / 1024
    return f"{size_kb / 1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"


def add_file(indent, filepath):
    filename = os.path.basename(filepath)
    ext      = os.path.splitext(filename)[1].lower()
    tag      = ext.upper().strip('.')
    lines.append(f"{indent}[{tag}]  {filename}  ({fmt_size(filepath)})")
    count_by_ext[ext] += 1


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Root directory files only (no recursion into subfolders)
# ══════════════════════════════════════════════════════════════════════════════
lines.append(f"[DIR]  {ROOT}/")

if os.path.isdir(ROOT):
    matched_root = sorted(
        f for f in os.listdir(ROOT)
        if os.path.isfile(os.path.join(ROOT, f))
        and os.path.splitext(f)[1].lower() in EXTENSIONS
    )
    if matched_root:
        for filename in matched_root:
            add_file("    ", os.path.join(ROOT, filename))
    else:
        lines.append("    (no matching files)")
else:
    lines.append(f"  [ERROR] Folder not found: {ROOT}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — asl_webapp folder (full recursive walk)
# ══════════════════════════════════════════════════════════════════════════════
lines.append("")

if not os.path.isdir(WEBAPP_DIR):
    lines.append(f"[SKIP]  asl_webapp not found at: {WEBAPP_DIR}")
else:
    for dirpath, dirnames, filenames in os.walk(WEBAPP_DIR):
        matched = sorted(
            f for f in filenames
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        )

        if not matched:
            continue  # skip folders with no matching files

        depth      = dirpath.replace(WEBAPP_DIR, "").count(os.sep)
        indent     = "    " * (depth + 1)      # +1 so it nests under ROOT
        sub_indent = "    " * (depth + 2)

        folder_name = os.path.basename(dirpath) if depth > 0 else "asl_webapp"
        lines.append(f"{indent}[DIR]  {folder_name}/")

        for filename in matched:
            add_file(sub_indent, os.path.join(dirpath, filename))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
summary = [
    "",
    "=" * 60,
    "  SUMMARY",
    "=" * 60,
]
for ext, count in sorted(count_by_ext.items()):
    summary.append(f"  {ext:<8} : {count} file(s)")
summary.append(f"\n  Total    : {sum(count_by_ext.values())} file(s)")
summary.append("=" * 60)

# ── Print ─────────────────────────────────────────────────────────────────────
header = [
    f"Scanning root       : {ROOT}",
    f"Scanning subfolder  : {WEBAPP_DIR}",
    f"Showing extensions  : {', '.join(sorted(EXTENSIONS))}",
    "=" * 60,
    "",
]
print("\n".join(header))
print("\n".join(lines))
print("\n".join(summary))

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(header) + "\n")
    f.write("\n".join(lines) + "\n")
    f.write("\n".join(summary) + "\n")

print(f"\n  Saved → {OUTPUT}")
