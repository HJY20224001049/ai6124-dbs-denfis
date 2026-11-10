"""Build an integration notebook and five-slide method draft from local artifacts.

Optional dependencies: requirements-deliverables.txt. No experiment is rerun.
The notebook embeds the actual model source and executes a synthetic contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    nbf = nbformat.v4
    report = (ROOT / "writing/A_method.md").read_text()
    notebook = nbf.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python 3 (method)",
                                   "language": "python", "name": "python3"}
    notebook.cells = [nbf.new_markdown_cell(
        "# DENFIS method integration draft\n\nLIN Jingzhao / HU Junyu. "
        "Not frozen; merge these cells with the data/evaluation sections. "
        "Runs offline from either repository root or writing/. No holdout evaluation."),
        nbf.new_code_cell(
            "from pathlib import Path\nimport os, sys\n"
            "root = Path.cwd()\n"
            "if not (root / 'config.py').exists():\n    root = root.parent\n"
            "assert (root / 'config.py').exists()\nos.chdir(root)\n"
            "sys.path.insert(0, str(root))\nimport numpy as np\nimport pandas as pd\n"
            "import matplotlib.pyplot as plt\nprint('Local data:', sorted(p.name for p in (root/'data/raw').glob('*.csv')))"),
        nbf.new_markdown_cell(report),
        nbf.new_markdown_cell("## Complete core implementation\n\n"
                              "The source below is embedded for review, not hidden behind an import."),
        nbf.new_code_cell((ROOT / "src/models/denfis.py").read_text()),
        nbf.new_code_cell(
            "rng = np.random.default_rng(42)\nX = rng.normal(size=(500, 2))\n"
            "y = np.sin(X[:, 0]) + 0.5*X[:, 1]\n"
            "m = DENFIS(dthr=.4, lam=1, prediction_clip=None)\n"
            "m.fit(X[:400], y[:400])\npred = m.predict(X[400:])\n"
            "assert np.mean((pred-y[400:])**2) < .3*np.mean(y[400:]**2)\n"
            "print('Synthetic test RMSE:', np.sqrt(np.mean((pred-y[400:])**2)), 'rules:', m.n_rules_)"),
        nbf.new_markdown_cell("## Saved out-of-sample results\n\n"
                              "Main and online tables are separate. Loading artifacts does not rerun training."),
        nbf.new_code_cell(
            "summary = pd.read_csv('results/summary.csv')\n"
            "pooled = summary[summary.scope == 'pooled']\n"
            "print('Frozen main comparison')\n"
            "print(pooled[pooled.model.isin(['denfis_frozen','xgboost','lstm'])].to_string(index=False))\n"
            "print('Online extension')\nprint(pooled[pooled.model == 'denfis_online'].to_string(index=False))"),
        nbf.new_code_cell(
            "for name in ['denfis_sensitivity.png', 'denfis_rule_evolution.png',"
            " 'denfis_ablations.png', 'denfis_memberships.png', 'denfis_case_contributions.png']:\n"
            "    path = root/'results/figures'/name\n"
            "    assert path.exists(), path\n"
            "    fig, ax = plt.subplots(figsize=(9,4))\n"
            "    ax.imshow(plt.imread(path)); ax.axis('off'); plt.show()"),
        nbf.new_markdown_cell("## Transfer extension\n\nThree aligned folds; "
                              "separate market scalers. Do not pool these rows into nine-fold main results."),
        nbf.new_code_cell("transfer = pd.read_csv('results/transfer/summary.csv')\n"
                          "print(transfer[['protocol','fold_id','rmse','da','sharpe']].to_string(index=False))"),
        nbf.new_markdown_cell((ROOT / "docs/refs.md").read_text()),
    ]
    path = ROOT / "writing/A_method.ipynb"
    nbformat.write(notebook, path)
    print(path)


def build_slides():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    pooled = pd.read_csv(ROOT / "results/summary.csv")
    row = pooled[(pooled.model == 'denfis_frozen') & (pooled.scope == 'pooled')].iloc[0]
    slides = [
        ("DBS next-day returns: evolving local rules", [
            "LIN Jingzhao | method section | preliminary, not frozen",
            "13 causal features; nine chronological test years (2017–2025)",
            "Training-only preprocessing; 2026 DENFIS holdout remains reserved",
            "Question: can an interpretable local model add forecast or trading value?"], None),
        ("ECM → Gaussian rules → first-order TSK", [
            "Normalised centre distance; threshold controls new regions",
            "At most 200 rules; optional support-based merging",
            "Two-pass batch fit: antecedents first, then local weighted RLS",
            "A DENFIS-inspired implementation; original-paper differences disclosed"], 'docs/figs/ecm_demo.png'),
        ("Frozen main comparison; online is a separate extension", [
            "Predict → record → realised-label update; predict never changes state",
            "20 Dthr × lambda settings selected using validation RMSE",
            "Final refit includes validation; thresholds selected before test",
            "Nine-fold default explicitly excludes the holdout"], 'results/figures/denfis_rule_evolution.png'),
        ("Forecast error and trading outcomes tell different stories", [
            f"Frozen: RMSE {row.rmse:.6f}; DA {row.da:.1%}; Sharpe {row.sharpe:.3f}",
            f"Maximum drawdown {row.max_dd:.1%}; exposure {row.exposure:.1%}",
            "No lower pooled RMSE than the historical baselines",
            "Sharpe point estimate below buy-and-hold; discuss statistical uncertainty"], 'results/figures/baseline_equity.png'),
        ("What the rules explain — and what remains uncertain", [
            "Memberships and local contributions expose the numerical prediction",
            "Three-fold transfer and inherited-parameter ablations are exploratory",
            "Single asset, regime sensitivity and bounded approximations limit claims",
            "Next: consultation, joint freeze, full notebook integration and rehearsal"], 'results/figures/denfis_case_contributions.png'),
    ]
    for index, (title, points, picture) in enumerate(slides, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(.5), Inches(.35), Inches(12.3), Inches(.8))
        p = box.text_frame.paragraphs[0]
        p.text, p.font.size, p.font.bold = title, Pt(27), True
        p.font.color.rgb = RGBColor.from_string('1F4E79')
        width = 5.6 if picture else 11.8
        box = slide.shapes.add_textbox(Inches(.6), Inches(1.55), Inches(width), Inches(4.8))
        for j, point in enumerate(points):
            p = box.text_frame.paragraphs[0] if j == 0 else box.text_frame.add_paragraph()
            p.text, p.font.size, p.space_after = point, Pt(21), Pt(22)
        if picture:
            path = ROOT / picture
            assert path.exists(), path
            # Fit image into the right column while retaining aspect ratio.
            from PIL import Image
            with Image.open(path) as img:
                aspect = img.width / img.height
            w, h = 6.3, 4.5
            if w / h > aspect:
                w = h * aspect
            else:
                h = w / aspect
            slide.shapes.add_picture(str(path), Inches(6.6), Inches(1.8), width=Inches(w), height=Inches(h))
        footer = slide.shapes.add_textbox(Inches(.6), Inches(7), Inches(11), Inches(.3))
        footer.text_frame.text = f"AI6124 Group 18 | Draft | {index}/5 | Sources: docs/refs.md"
        footer.text_frame.paragraphs[0].font.size = Pt(11)
        slide.notes_slide.notes_text_frame.text = (
            "Target approximately 55 seconds. Explain assumptions and uncertainty; "
            "do not imply instructor approval or guaranteed predictive advantage. "
            "Use writing/A_method.md for the detailed account."
        )
    path = ROOT / 'writing/A_method_slides.pptx'
    prs.save(path)
    assert len(Presentation(path).slides) == 5
    print(path)


if __name__ == '__main__':
    build_notebook()
    build_slides()
