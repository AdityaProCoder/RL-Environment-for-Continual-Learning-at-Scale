# ==============================================================
# kaggle_pkg.py  — build a self-contained gcl package zip for the notebook
# Run once:  python kaggle_pkg.py
# Produces:  gcl/gcl_pkg_b64.txt  (base64 of a zip with all gcl/*.py inside)
# ==============================================================
import os, io, zipfile, base64, glob

GCL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcl")

files = []
for pat in ["*.py", "learners/*.py"]:
    for fp in glob.glob(os.path.join(GCL_DIR, pat), recursive=True):
        rel = os.path.relpath(fp, GCL_DIR).replace("\\", "/")
        files.append((rel, fp))

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for rel, fp in files:
        # Prefix with "gcl/" so the zip extracts as gcl/config.py, gcl/engine.py, ...
        z.writestr(f"gcl/{rel}", open(fp, "rb").read())
buf.seek(0)
b64 = base64.b64encode(buf.read()).decode("ascii")
out = os.path.join(GCL_DIR, "gcl_pkg_b64.txt")
with open(out, "w") as f:
    f.write(b64)
print("wrote", out, "->", len(b64), "chars,", len(files), "files")
