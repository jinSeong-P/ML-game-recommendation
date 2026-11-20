"""Regression checks for v2 experiment allowlists and frozen-selection locks."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import run_improvement_v2 as runner
from eda_large_review_frank import digest


class ImprovementRunnerTests(unittest.TestCase):
    def test_variant_set_requires_exact_membership_and_unique_rows(self):
        expected = {"a", "b"}
        runner.require_variant_set([{"variant": "a"}, {"variant": "b"}], expected)
        for values in (
            [{"variant": "a"}],
            [{"variant": "a"}, {"variant": "b"}, {"variant": "extra"}],
            [{"variant": "a"}, {"variant": "a"}],
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "Predeclared variants mismatch"):
                runner.require_variant_set(values, expected)

    def test_text_variants_follow_t0_gate_not_artifact_presence(self):
        with tempfile.TemporaryDirectory() as folder:
            eda = Path(folder)
            report = eda / "reports/tfidf_ranking_analysis_v2.json"
            report.parent.mkdir(parents=True)
            (eda / "data/processed/modeling/improvement_v2/tfidf/T3_max_df_0_5").mkdir(parents=True)
            for choice, has_t3 in (("skip", False), ("build", True)):
                report.write_text(json.dumps({"diagnostics": {"T3_conditional_gate": {"t3_choice": choice}}}), encoding="utf-8")
                with mock.patch.object(runner, "EDA", eda):
                    variants = runner.expected_text_variants()
                self.assertEqual("text_T3_max_df_0_5" in variants, has_t3)

    def test_freeze_rejects_missing_or_extra_validation_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(runner, "ROOT", root), \
                 mock.patch.object(runner, "expected_validation_variants", return_value={"a", "b"}):
                for values in ([{"variant": "a"}],
                               [{"variant": "a"}, {"variant": "b"}, {"variant": "extra"}]):
                    with self.subTest(values=values), \
                         mock.patch.object(runner, "summaries", return_value=values), \
                         self.assertRaisesRegex(ValueError, "Predeclared variants mismatch"):
                        runner.freeze_selection()

    def test_text_one_cli_rejects_unlisted_name_and_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            with mock.patch.object(runner, "ROOT", root), \
                 mock.patch.object(runner, "expected_text_variants", return_value={"text_T1_sublinear_tf"}), \
                 mock.patch.object(runner, "evaluate_text_directory") as evaluate:
                cases = [
                    ["run_improvement_v2.py", "text-one", "--name", "text_custom", "--path", str(root / "tfidf/custom")],
                    ["run_improvement_v2.py", "text-one", "--name", "text_T1_sublinear_tf", "--path", str(root / "elsewhere")],
                ]
                for argv in cases:
                    with self.subTest(argv=argv), mock.patch("sys.argv", argv), self.assertRaises(ValueError):
                        runner.main()
                evaluate.assert_not_called()

                expected = root / "tfidf/T1_sublinear_tf"
                with mock.patch("sys.argv", ["run_improvement_v2.py", "text-one", "--name", "text_T1_sublinear_tf", "--path", str(expected)]):
                    runner.main()
                evaluate.assert_called_once_with(expected.resolve(), "text_T1_sublinear_tf")
                evaluate.reset_mock()

                relative = os.path.relpath(expected, Path.cwd())
                with mock.patch("sys.argv", ["run_improvement_v2.py", "text-one", "--name", "text_T1_sublinear_tf", "--path", relative]):
                    runner.main()
                evaluate.assert_called_once_with(expected.resolve(), "text_T1_sublinear_tf")

    def test_metadata_one_rejects_absent_and_unplanned_alpha(self):
        with mock.patch.object(runner, "run_metadata") as run:
            with mock.patch("sys.argv", ["run_improvement_v2.py", "metadata-one"]), \
                 self.assertRaisesRegex(ValueError, "metadata-one requires a predeclared alpha"):
                runner.main()
            run.assert_not_called()

            with mock.patch("sys.argv", ["run_improvement_v2.py", "metadata-one", "--alpha", "0.4"]), \
                 redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_metadata_one_dispatches_each_allowed_alpha_only(self):
        for alpha in (1.0, 0.75, 0.5, 0.25):
            with self.subTest(alpha=alpha), mock.patch.object(runner, "run_metadata") as run, \
                 mock.patch("sys.argv", ["run_improvement_v2.py", "metadata-one", "--alpha", str(alpha)]):
                runner.main()
                run.assert_called_once_with([alpha])

    def test_expected_text_variants_require_the_t0_decision(self):
        with tempfile.TemporaryDirectory() as folder:
            eda = Path(folder)
            with mock.patch.object(runner, "EDA", eda):
                with self.assertRaises(FileNotFoundError):
                    runner.expected_text_variants()

    def test_selected_model_manifest_and_file_hashes_are_rechecked(self):
        with tempfile.TemporaryDirectory() as folder:
            eda = Path(folder)
            model = eda / "data/models/text"
            model.mkdir(parents=True)
            weights = model / "weights.bin"
            weights.write_bytes(b"model-v1")
            manifest = model / "manifest.json"
            manifest.write_text(json.dumps({"files": {"weights.bin": {"sha256": digest(weights)}}}), encoding="utf-8")
            spec = {"directory": "data/models/text", "manifest_sha256": digest(manifest)}

            with mock.patch.object(runner, "EDA", eda):
                runner.verify_model_sources(spec)

                weights.write_bytes(b"changed weights")
                with self.assertRaisesRegex(ValueError, "Model output changed"):
                    runner.verify_model_sources(spec)

                weights.write_bytes(b"model-v1")
                manifest.write_text(json.dumps({"files": {"weights.bin": {"sha256": digest(weights)}}, "note": "changed"}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Model manifest changed"):
                    runner.verify_model_sources(spec)

    def _write_selection_fixture(self, root, published, protocol, *, runner_sha=None, evidence_sha=None, model=None):
        validation_manifest = root / "evaluations/validation/only/manifest.json"
        validation_manifest.parent.mkdir(parents=True)
        validation_manifest.write_text("validation", encoding="utf-8")
        result = {
            "version": "improvement-v2-selection",
            "status": "frozen_before_reference_test",
            "protocol_sha256": digest(protocol),
            "runner_sha256": runner_sha if runner_sha is not None else digest(runner.__file__),
            "models": {} if model is None else {"tfidf": {"model": model}},
            "validation_evidence": {"only": evidence_sha if evidence_sha is not None else digest(validation_manifest)},
        }
        selection = root / "selection_frozen.json"
        selection.parent.mkdir(parents=True, exist_ok=True)
        selection.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_bytes(selection.read_bytes())
        return result, selection, validation_manifest

    def _load_selection(self, root, published, protocol, eda=None):
        return mock.patch.multiple(
            runner,
            ROOT=root,
            EDA=runner.EDA if eda is None else eda,
            PUBLISHED_SELECTION=published,
            PROTOCOL=protocol,
            expected_validation_variants=mock.DEFAULT,
            summaries=mock.DEFAULT,
            verify_validation_evidence=mock.DEFAULT,
        )

    def test_reference_gate_rejects_selection_snapshot_and_runner_mismatches(self):
        for mismatch in ("snapshot", "runner"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "run"
                published = Path(folder) / "reference/improvement_selection_v2.json"
                protocol = Path(folder) / "protocol.json"
                protocol.write_text("{}", encoding="utf-8")
                bad_runner = "0" * 64 if mismatch == "runner" else None
                self._write_selection_fixture(root, published, protocol, runner_sha=bad_runner)
                if mismatch == "snapshot":
                    published.write_bytes(published.read_bytes() + b" ")

                with self._load_selection(root, published, protocol) as patched:
                    patched["expected_validation_variants"].return_value = {"only"}
                    patched["summaries"].return_value = [{"variant": "only"}]
                    with self.assertRaisesRegex(ValueError, "Frozen selection differs from published snapshot|Frozen selection protocol/runner changed"):
                        runner.load_verified_selection()

    def test_reference_gate_rejects_changed_validation_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            published = Path(folder) / "reference/improvement_selection_v2.json"
            protocol = Path(folder) / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            self._write_selection_fixture(root, published, protocol, evidence_sha="f" * 64)
            with self._load_selection(root, published, protocol) as patched:
                patched["expected_validation_variants"].return_value = {"only"}
                patched["summaries"].return_value = [{"variant": "only"}]
                with self.assertRaisesRegex(ValueError, "Frozen validation evidence changed"):
                    runner.load_verified_selection()

    def test_reference_gate_rechecks_selected_model_manifest_and_files(self):
        for mutation in ("manifest", "file"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "run"
                eda = Path(folder) / "eda"
                published = eda / "data/reference/improvement_selection_v2.json"
                protocol = Path(folder) / "protocol.json"
                protocol.write_text("{}", encoding="utf-8")
                model_dir = eda / "data/models/text"
                model_dir.mkdir(parents=True)
                weights = model_dir / "weights.bin"
                weights.write_bytes(b"model-v1")
                manifest = model_dir / "manifest.json"
                manifest.write_text(json.dumps({"files": {"weights.bin": {"sha256": digest(weights)}}}), encoding="utf-8")
                spec = {"directory": "data/models/text", "manifest_sha256": digest(manifest)}
                self._write_selection_fixture(root, published, protocol, model=spec)
                if mutation == "file":
                    weights.write_bytes(b"changed weights")
                else:
                    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")

                with self._load_selection(root, published, protocol, eda=eda) as patched:
                    patched["expected_validation_variants"].return_value = {"only"}
                    patched["summaries"].return_value = [{"variant": "only"}]
                    with self.assertRaisesRegex(ValueError, "Model (manifest changed|output changed)"):
                        runner.load_verified_selection()


if __name__ == "__main__":
    unittest.main()
