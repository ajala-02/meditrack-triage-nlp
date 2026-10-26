import unittest

from meditrack_triage.baseline import top_features, train_baseline
from meditrack_triage.data import generate_synthetic_dataset


class BaselineTests(unittest.TestCase):
    def test_baseline_trains_and_returns_probabilities(self):
        dataset = generate_synthetic_dataset(rows_per_label=20, seed=3)
        model = train_baseline(dataset)
        probabilities = model.predict_proba(["I have crushing pressure in my chest"])
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertEqual(set(top_features(model)), {"low", "medium", "high"})


if __name__ == "__main__":
    unittest.main()

