import os, sys

ROOT = "."
EXCLUDE_DIRS = {
    ".git",".venv","__pycache__","node_modules","dist","build","static",
    ".ipynb_checkpoints","Lib","Include","site-packages",".mypy_cache"
}

# Folders to collapse (show head/tail and a count)
COLLAPSE_DIRS = {"data/incoming","data/processed"}
HEAD, TAIL = 3, 2

ALLOWED_EXT = {".py",".ipynb",".md",".yml",".yaml",".sql",".csv"}
ALLOWED_NAMES = {"Dockerfile","docker-compose.yml","LICENSE",".gitignore",".gitkeep",
                 "requirements.txt","README.md"}

def want_dir(name): return name not in EXCLUDE_DIRS

def want_file(path):
    base = os.path.basename(path)
    if base in ALLOWED_NAMES:
        return True
    _, ext = os.path.splitext(base)
    return ext in ALLOWED_EXT

def list_filtered(d):
    try:
        items = sorted(os.listdir(d), key=str.lower)
    except PermissionError:
        return [], []
    dirs, files = [], []
    for x in items:
        p = os.path.join(d, x)
        if os.path.isdir(p):
            if want_dir(x): dirs.append(x)
        else:
            if want_file(p): files.append(x)
    return dirs, files

def draw(d, prefix=""):
    rel = os.path.relpath(d, ROOT).replace("\\","/")
    dirs, files = list_filtered(d)

    if rel in COLLAPSE_DIRS and len(files) > HEAD + TAIL + 1:
        shown = files[:HEAD] + files[-TAIL:]
        hidden = len(files) - len(shown)
        files = shown + [f"... ({hidden} more files)"]

    items = [(True, x) for x in dirs] + [(False, x) for x in files]
    for i, (is_dir, name) in enumerate(items):
        last = (i == len(items) - 1)
        branch = "+-- "
        print(prefix + branch + name)
        if is_dir:
            draw(os.path.join(d, name), prefix + ("   " if last else "|  "))

top = os.path.basename(os.path.abspath(ROOT))
print(f"{top}/")
draw(ROOT)
