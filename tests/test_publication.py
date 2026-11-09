"""Check leak detection and the boundaries that matter to the public claims."""
import importlib.util
import json
from pathlib import Path
import unittest
import ast
import os
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

class PublicationTests(unittest.TestCase):
    def test_private_identifiers_are_detected(self):
        check=module('validate_publication').findings
        self.assertIn('email',check('example'+'@'+'private.invalid'))
        self.assertIn('github_token',check('ghp_'+'a'*40))
        self.assertIn('private_drive_link',check('https://'+'drive.google.com/'+'file/d/example'))
    def test_cost_boundaries_are_not_double_counted(self):
        result,_,_=module('analyze').compute()
        startup=result['startup_observation']
        self.assertAlmostEqual(startup['pre_request_preparation_seconds']+startup['first_request_with_validation_seconds'],startup['time_to_first_validated_media_seconds'])
        self.assertAlmostEqual(startup['preparation_cu_estimate']+startup['first_request_cu_estimate'],startup['time_to_first_media_cu_estimate'])
    def test_unexecuted_tests_stay_unexecuted(self):
        result=json.loads((ROOT/'data/results.json').read_text())
        self.assertEqual(result['accepted_main_clips'],0)
        self.assertFalse(result['formal_matrix_completed'])
        self.assertFalse(result['large_scale_batch_or_concurrency_validated'])
    def test_public_files_pass_review_rules(self):
        self.assertGreater(len(module('validate_publication').validate()),20)
    def test_public_allocation_stops_before_any_account_access(self):
        source=ROOT/'projects/minimax-h3/study/control.py'
        tree=ast.parse(source.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='allocate')
        namespace={'os':os}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'public-allocation-opt-in','exec'),namespace)
        with patch.dict(os.environ,{'COLAB_MEDIA_ALLOW_GPU':'0'}):
            with self.assertRaisesRegex(RuntimeError,'GPU allocation disabled'):
                namespace['allocate']('research')

if __name__=='__main__': unittest.main()
