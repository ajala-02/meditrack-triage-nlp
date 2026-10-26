import unittest

from meditrack_triage.constants import LABELS
from meditrack_triage.data import create_splits, generate_synthetic_dataset


class SyntheticDataTests(unittest.TestCase):
    def test_generation_is_balanced_and_synthetic(self):
        dataset = generate_synthetic_dataset(rows_per_label=20, seed=7)
        self.assertEqual(len(dataset), 60)
        self.assertEqual(dataset["source"].unique().tolist(), ["synthetic"])
        self.assertEqual(set(dataset["triage"]), set(LABELS))
        self.assertTrue((dataset.groupby("triage").size() == 20).all())

    def test_splits_are_disjoint_and_stratified(self):
        dataset = generate_synthetic_dataset(rows_per_label=30, seed=7)
        splits = create_splits(dataset, seed=7)
        all_ids = set().union(*(set(part["id"]) for part in splits.values()))
        self.assertEqual(len(all_ids), len(dataset))
        for part in splits.values():
            self.assertEqual(set(part["triage"]), set(LABELS))


if __name__ == "__main__":
    unittest.main()

