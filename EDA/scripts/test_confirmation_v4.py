"""Prevent outcome opening for stale or expanded finalist locks."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from evaluate_confirmation_v4 import validate_lock, run


class ConfirmationV4Tests(unittest.TestCase):
    def setUp(self):
        self.lock = dict(status='frozen_after_selection_before_confirmation',
                         protocol_sha256='p', split_manifest_sha256='s', execution_plan_sha256='e',
                         finalist='rlae_final', confirmation_models=[
                             'activity_inverse_degree_cosine_k100', 'binary_item_knn_k100',
                             'metadata_extended_tfidf', 'rlae_final'])

    def validate(self, lock):
        with patch('evaluate_confirmation_v4.digest', side_effect=['p', 's', 'e']):
            validate_lock(lock)

    def test_exactly_one_challenger_or_no_opening(self):
        self.validate(self.lock)
        self.validate(dict(self.lock, finalist=None, confirmation_models=[]))
        with self.assertRaises(ValueError):
            self.validate(dict(self.lock, confirmation_models=self.lock['confirmation_models'] + ['another']))

    def test_no_finalist_cannot_open_controls_for_fishing(self):
        with self.assertRaises(ValueError):
            self.validate(dict(self.lock, finalist=None))

    def test_changed_protocol_and_unfrozen_selection_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(dict(self.lock, protocol_sha256='stale'))
        with self.assertRaises(ValueError):
            validate_lock(dict(self.lock, status='running'))

    def test_execution_schedule_cannot_change_after_freeze(self):
        with self.assertRaises(ValueError):
            self.validate(dict(self.lock, execution_plan_sha256='changed'))

    def test_scorer_failure_occurs_before_reserved_targets_are_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'selection_frozen.json').write_text(json.dumps(self.lock),encoding='utf-8')
            with patch('evaluate_confirmation_v4.ROOT',root), patch('evaluate_confirmation_v4.EDA',root), \
                patch('evaluate_confirmation_v4.validate_lock'), \
                patch('evaluate_confirmation_v4.restore_selected',side_effect=ValueError('scorer mismatch')), \
                patch('evaluate_confirmation_v4.load_cohort') as load:
                with self.assertRaisesRegex(ValueError,'scorer mismatch'):run()
                load.assert_not_called()

    def test_changed_lock_cannot_start_a_second_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'reports').mkdir()
            (root/'selection_frozen.json').write_text(json.dumps(self.lock),encoding='utf-8')
            (root/'reports/confirmation_v4.json').write_text(json.dumps({'selection_lock_sha256':'old'}),encoding='utf-8')
            with patch('evaluate_confirmation_v4.ROOT',root), patch('evaluate_confirmation_v4.EDA',root), \
                patch('evaluate_confirmation_v4.validate_lock'), patch('evaluate_confirmation_v4.load_cohort') as load:
                with self.assertRaisesRegex(ValueError,'different selection lock'):run()
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
