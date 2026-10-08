"""Same 6-panel explanatory figure as hybrid_s1_dvf_warped_target.png, for every exported model (reads the .mha files)."""
import sys, os, csv, numpy as np, SimpleITK as sitk, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
M = Path(sys.argv[1]); rd = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
tre = {r["model"]: r for r in csv.DictReader(open(M / "tre_summary.csv"))}
cases = (1, 4, 8); com = {c: {k: rd(M / "common" / f"C{c:02d}_{k}.mha") for k in ("input_CT06_T50", "target_CT01_T00", "dvf_elastix_mm")} for c in cases}
for d in sorted(p for p in M.iterdir() if p.is_dir() and p.name != "common"):
    if "push" in d.name:
        (d / "NOTE.txt").write_text("Push labels are expressed on the input (reference) grid, not the target grid. The standard pull warp\n"
                                    "used here does not apply, so these volumes and the 14.9 mm TRE are NOT valid. Its correct score was 4.38 mm\n"
                                    "(Diary 2026-10-01 22:40). No figure made.\n"); continue
    fig, ax = plt.subplots(len(cases), 6, figsize=(24, 4.4 * len(cases)))
    for i, c in enumerate(cases):
        inp, tgt, el = com[c]["input_CT06_T50"], com[c]["target_CT01_T00"], com[c]["dvf_elastix_mm"]
        w, u = rd(d / f"C{c:02d}_warped_CT06.mha"), rd(d / f"C{c:02d}_dvf_mm.mha")
        lung = inp < -400; z = int(np.round(np.where(lung)[0].mean()))
        kw = dict(cmap="gray", vmin=-1000, vmax=300)
        ax[i, 0].imshow(inp[z], **kw); ax[i, 0].set_title(f"C{c:02d} input CT_06 (T50, exhale)")
        ax[i, 1].imshow(tgt[z], **kw); ax[i, 1].set_title("target CT_01 (T00, inhale)")
        ax[i, 2].imshow(w[z], **kw); ax[i, 2].set_title(f"warped by model (TRE {float(tre[d.name][f'C{c:02d}']):.2f} mm)")
        ax[i, 3].imshow(tgt[z] - inp[z], cmap="bwr", vmin=-600, vmax=600); ax[i, 3].set_title("target - input (before)")
        ax[i, 4].imshow(tgt[z] - w[z], cmap="bwr", vmin=-600, vmax=600); ax[i, 4].set_title("target - warped (after)")
        mag = np.linalg.norm(u[z], axis=-1); emag = np.linalg.norm(el[z], axis=-1)
        ax[i, 5].imshow(mag, cmap="magma", vmin=0, vmax=max(emag.max(), mag.max(), 1)); st = 6; yy, xx = np.mgrid[0:160:st, 0:160:st]
        ax[i, 5].quiver(xx, yy, u[z, ::st, ::st, 0], u[z, ::st, ::st, 1], color="cyan", angles="xy", scale_units="xy", scale=0.5, width=0.003, label="model")
        ax[i, 5].quiver(xx, yy, el[z, ::st, ::st, 0], el[z, ::st, ::st, 1], color="lime", angles="xy", scale_units="xy", scale=0.5, width=0.002, alpha=.6, label="Elastix")
        ax[i, 5].set_title(f"model |DVF| max {mag.max():.1f} mm; Elastix max {emag.max():.1f}"); ax[i, 5].legend(loc="lower right", fontsize=8)
        for a in ax[i]: a.axis("off")
    plt.suptitle(f"{d.name} + mirror (DIR-Lab mean TRE300 {float(tre[d.name]['mean']):.2f} mm) — coronal slice through lung centre, head at top", fontsize=14)
    plt.tight_layout(); plt.savefig(d / "explanatory_figure.png", dpi=90); plt.close(fig); print("figure", d.name, flush=True)
