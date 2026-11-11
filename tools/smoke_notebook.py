"""CPU smoke test of notebooks/pipeline.ipynb on 2 val + 2 test images (does not touch the GPU).

  CUDA_VISIBLE_DEVICES= python tools/smoke_notebook.py <weights.pt> <pp.json>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import nbformat
import torch

torch.set_num_threads(4)
weights, pp = sys.argv[1], sys.argv[2]
nb = nbformat.read("notebooks/pipeline.ipynb", as_version=4)
src = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
src = (src.replace('"weights/maskrcnn_r50_filament.pt"', repr(weights))
          .replace('"weights/maskrcnn_r50_filament.pp.json"', repr(pp))
          .replace('"submissions/submission.csv"', '"submissions/_smoke.csv"')
          .replace("vs = ValSet(cfg)", "vs = ValSet(cfg); vs.paths = vs.paths[:2]; _t = {p.stem for p in vs.paths}; "
                   "vs.samples = [s for s in vs.samples if s['timestamp'] in _t]")
          .replace('test_paths = sorted(test_dir.glob("*.jp*g"))', 'test_paths = sorted(test_dir.glob("*.jp*g"))[:2]'))
exec(compile(src, "pipeline.ipynb", "exec"), {"display": print, "__name__": "__main__"})
print("NOTEBOOK SMOKE OK")
