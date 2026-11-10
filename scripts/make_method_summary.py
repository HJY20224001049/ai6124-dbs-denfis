"""Render saved transfer and ablation results for notebook integration."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_results_table import markdown_table


def main():
    transfer = pd.read_csv(ROOT/'results/transfer/summary.csv')
    ablation = pd.read_csv(ROOT/'results/analysis/ablations.csv')
    text = [
        '# Method extensions — preliminary results', '',
        'These exploratory experiments are separate from the nine-fold frozen main comparison. No DENFIS holdout was evaluated.', '',
        '## STI to DBS transfer', '',
        'Aligned label dates and separate market training scalers. Strict zero-shot uses STI validation labels for both hyperparameters and the trading threshold; scratch/fine-tune use DBS validation. Every variant is frozen during target test evaluation.', '',
        markdown_table(transfer[['protocol','fold_id','n','rmse','da','sharpe','n_rules']]), '',
        'Fine-tuning has lower RMSE than aligned scratch in all three sampled folds, but these are only three exploratory folds. Trading improvements are not uniform. Do not compare these aligned-sample scores as if their rows were identical to the full DBS nine-fold run.', '',
        '## Ablations', '',
        'All ablations inherit each frozen model\'s selected parameters. no_rls uses weighted constants, changing consequent complexity as well as the update. r2f1 is added because it actually exercises merging.', '',
        markdown_table(ablation), '',
        'The simpler constant variant sometimes has lower error. Disabling merge is identical in the planned three representative fits because no merge occurred; the merge-active r2f1 changes 66 rules to 67 with only a small error change.', '',
        '## Figures and interpretation', '',
        '- `results/figures/denfis_sensitivity.png`: validation profile minima over the other grid parameter, not test-driven tuning.',
        '- `results/figures/denfis_rule_evolution.png`: online rule counts after realised labels.',
        '- `results/figures/denfis_ablations.png`: inherited-parameter test scores.',
        '- `results/figures/denfis_memberships.png`: marginal coordinate memberships of the three most active r3f3 rules.',
        '- `results/figures/denfis_case_contributions.png`: post-hoc high-volatility example; no causal economic interpretation.', '',
        'Candidate outputs remain subject to consultation and the scheduled joint freeze. See docs/freeze.md.',
    ]
    path = ROOT/'docs/method_extensions.md'
    path.write_text('\n'.join(text)+'\n')
    print(path)


if __name__ == '__main__':
    main()
