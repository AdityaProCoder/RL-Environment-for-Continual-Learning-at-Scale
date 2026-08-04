import json, base64, zipfile, io, os, sys, shutil
nb = json.load(open(r"D:/Project/continuos env/kaggle_continual_learning.ipynb"))
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if 'B64 = "' in src and "zipfile" in src:
        line = [l for l in src.splitlines() if "B64 = " in l][0]
        b64 = line.split('"',1)[1].rsplit('"',1)[0]
        out = os.path.expanduser("./_nbtest")
        shutil.rmtree(out, ignore_errors=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))) as z:
            z.extractall(out)
        print("extracted", len(z.namelist()), "files ->", out)
        print("top-level:", os.listdir(out))
        sys.path.insert(0, out)
        import gcl
        print("IMPORT_OK:", gcl.__version__, "| learners:", sorted(gcl.LEARNERS))
        break
