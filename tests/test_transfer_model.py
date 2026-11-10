"""Transfer training boundaries and source/target label use."""
import numpy as np

from scripts.run_transfer import TransferDENFIS


def test_zero_shot_ignores_target_labels_and_finetune_uses_them():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 2))
    source = {50: (X, np.zeros(50))}
    models = []
    for protocol, label in [("zero_shot", 0.), ("zero_shot", .05), ("fine_tune", .05)]:
        m = TransferDENFIS(source, protocol, lam=1)
        m.fit(X, np.full(50, label))
        models.append(m)
    np.testing.assert_array_equal(models[0].predict(X), models[1].predict(X))
    assert np.mean(models[2].predict(X)) > 0.01
    assert models[2].evolve_rules is False
