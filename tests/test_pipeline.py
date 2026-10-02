from __future__ import annotations

import tempfile
import unittest
import wave
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.extract_audio_features import (ROOT, EGeMAPSExtractor, TransformerEmbeddingExtractor,
    _get_features, atomic_json, derive_timing, extract_features, load_config, load_split_data)
from scripts.compare_audio_models import (available_models, build_pipeline, candidate_params,
    estimator, load_feature_group, metrics, _assert_no_target_columns)


class FixedSplitTests(unittest.TestCase):
    def test_fixed_splits_are_disjoint_and_match_manifests(self):
        config = load_config()
        data = load_split_data(config)
        sets = {name: set(frame.user_id.astype(str)) for name, frame in data.items()}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])
        self.assertEqual([len(data[s]) for s in ("train", "val", "test")], [1394, 319, 298])

    def test_confidence_target_alias_is_removed_from_features(self):
        config = load_config()
        data = load_split_data(config)
        for branch in ("legacy", "egemaps"):
            features = load_feature_group(branch, data, config)
            self.assertNotIn("confidence", features["val"].columns)
            _assert_no_target_columns(features["val"], config, f"{branch} leakage regression test")

        # Recreate the bug shape: an extractor CSV's `confidence` alias carries
        # confidence_score even though the configured target column is named differently.
        leaked_X = features["val"].assign(
            confidence=pd.to_numeric(data["val"]["confidence_score"]).to_numpy())
        with self.assertRaises(AssertionError):
            _assert_no_target_columns(leaked_X, config, "deliberate leak")


class ExtractorShapeTests(unittest.TestCase):
    @staticmethod
    def _tone(path):
        sr = 16000
        x = (np.sin(2 * np.pi * 180 * np.arange(sr) / sr) * 0.2 * 32767).astype("<i2")
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(sr); stream.writeframes(x.tobytes())

    def test_waveform_feature_schema_and_f0(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tone.wav"
            self._tone(path)
            values = extract_features(path, use_transcript=False)
        required = {"f0_hz_mean", "f0_hz_std", "rms_energy_mean", "rms_energy_range",
                    "speech_duration_seconds", "pause_count", "jitter_local_proxy_percent",
                    "shimmer_local_proxy_db"}
        self.assertTrue(required.issubset(values))
        self.assertAlmostEqual(values["f0_hz_mean"], 180, delta=10)
        self.assertFalse(any(name.startswith("transcript_") for name in values))

    @unittest.skipUnless(find_spec("opensmile"), "optional opensmile dependency is not installed")
    def test_egemaps_has_stable_88_named_functionals(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tone.wav"
            self._tone(path)
            values = EGeMAPSExtractor(load_config()).extract(path)
        self.assertEqual(len(values), 88)
        self.assertTrue(all(isinstance(name, str) and name for name in values))

    def test_embedding_layer_selection_and_whisper_timing_shapes(self):
        pooled = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
        selected = TransformerEmbeddingExtractor.selected_vector(None, pooled, "last_n", 2)
        np.testing.assert_array_equal(selected, pooled[-2:].mean(axis=0))
        features, evidence = derive_timing(
            [{"word": "um", "start": 0.2, "end": 0.5}, {"word": "hello", "start": 1.1, "end": 1.4}],
            2.0, ["um"], [0.3, 0.7, 1.5])
        self.assertEqual(features["word_count"], 2)
        self.assertEqual(features["filler_count"], 1)
        self.assertAlmostEqual(evidence["longest_pause"]["duration"], 0.6)

    def test_cached_extractor_value_is_reused(self):
        config = load_config()
        first = pd.read_csv(ROOT / "Datasets" / "RecruitView Audio" / "train_audio_metadata.csv",
                            dtype={"id": str}).iloc[0]
        expected = {"cached_feature": 3.25}
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            config["paths"]["cache_dir"] = temp
            cache = Path(temp) / "egemaps" / "eGeMAPSv02_Functionals" / "train" / f"{first.id}.json"
            atomic_json(cache, expected)

            class MustNotRun:
                def extract(self, _audio):
                    raise AssertionError("cache miss unexpectedly reran the extractor")

            observed = _get_features("egemaps", MustNotRun(), "train", first, config,
                                     Path(temp), force=False)
            self.assertEqual(observed, expected)


class ModelingTests(unittest.TestCase):
    def test_embedding_ridge_is_scaled_stable_and_uses_safer_alpha_grid(self):
        X = pd.DataFrame({"signal": [0.0, 2.0, 4.0]})
        model = build_pipeline(X, "Ridge", {"alpha": 1.0}, 42, standardize=True, stable_ridge=True)
        self.assertEqual(model.named_steps["model"].solver, "lsqr")
        params = candidate_params("Ridge", load_config(), 42, stable_ridge=True)
        self.assertGreaterEqual(min(item["alpha"] for item in params), 0.1)
        self.assertEqual(available_models("embedding", has_embedding=True), ["Ridge"])
        # The handcrafted default tuning/solver path is intentionally unchanged.
        self.assertEqual(estimator("Ridge", {"alpha": 1.0}, 42).solver, "auto")
        self.assertEqual(min(item["alpha"] for item in candidate_params("Ridge", load_config(), 42)), 0.0001)
        tree_params = candidate_params("ExtraTrees", load_config(), 42, max_trials=20)
        self.assertTrue(all(isinstance(item["max_features"], (float, str)) for item in tree_params))
        self.assertTrue(all(item["max_features"] == "sqrt" or item["max_features"] <= 1.0 for item in tree_params))

    def test_scaler_is_fit_only_on_pipeline_training_rows(self):
        X = pd.DataFrame({"signal": [0.0, 2.0]})
        model = build_pipeline(X, "Ridge", {"alpha": 1.0}, 42, standardize=True)
        model.fit(X, np.array([0.0, 1.0]))
        fitted_scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
        self.assertAlmostEqual(float(fitted_scaler.mean_[0]), 1.0)
        # Large held-out values have not entered the fit.
        model.predict(pd.DataFrame({"signal": [10000.0]}))
        self.assertAlmostEqual(float(fitted_scaler.mean_[0]), 1.0)

    def test_ccc_and_metrics(self):
        score = metrics([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(score["ccc"], 1.0)
        self.assertAlmostEqual(score["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
