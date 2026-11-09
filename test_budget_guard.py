import types
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import budget_guard as guard

class BudgetGuardTests(unittest.TestCase):
    def setUp(self):
        self.job = {"notebook_uuid": "00000000-0000-4000-8000-000000000001",
                    "before": {"balance_cu": 200}, "started": 1000, "deadline": 1180}
        self.current = {"balance_cu": 199.99, "rate_cu_per_hour": 1.8}
        self.t4 = [types.SimpleNamespace(accelerator=types.SimpleNamespace(value="T4"))]

    def test_short_t4_job_allowed(self):
        self.assertIsNone(guard.stop_reason(self.job, self.current, self.t4, 1010))

    def test_unlimited_study_still_protects_balance_and_progress(self):
        self.job.update(deadline=None,last_progress=1000)
        self.job['before']['balance_cu']=197
        self.job['policy']=dict(guard.POLICY,gpu='G4',max_cu=None,max_seconds=None,
                               balance_floor_cu=10,stall_seconds=1800,max_rate_cu_per_hour=12)
        self.current.update(balance_cu=197,rate_cu_per_hour=10.05)
        cards=[types.SimpleNamespace(accelerator=types.SimpleNamespace(value='G4'))]
        self.assertIsNone(guard.stop_reason(self.job,self.current,cards,1010))
        self.assertIn('progress',guard.stop_reason(self.job,self.current,cards,2801))
        self.current['balance_cu']=10.01
        self.assertIn('balance',guard.stop_reason(self.job,self.current,cards,1010))

    def test_transport_retry_is_bounded_and_legacy_policy_still_fails_closed(self):
        self.assertIsNone(guard.monitor_retry_delay(self.job,1,1000,1001))
        self.job['policy']=dict(guard.POLICY,monitor_failure_grace_seconds=45,monitor_transport_retries=2)
        self.assertEqual(guard.monitor_retry_delay(self.job,1,1000,1001),15)
        self.assertEqual(guard.monitor_retry_delay(self.job,2,1000,1038),7)
        self.assertIsNone(guard.monitor_retry_delay(self.job,3,1000,1030))
        self.assertIsNone(guard.monitor_retry_delay(self.job,1,1000,1045))

    def test_time_limit_and_high_rate_stop(self):
        self.assertIsNotNone(guard.stop_reason(self.job, self.current, self.t4, 1181))
        self.current["rate_cu_per_hour"] = 6
        self.assertIsNotNone(guard.stop_reason(self.job, self.current, self.t4, 1001))

    def test_budget_stops_before_threshold_with_teardown_reserve(self):
        self.current["balance_cu"] = 199.755
        self.assertIsNotNone(guard.stop_reason(self.job, self.current, self.t4, 1010))

    def test_wrong_gpu_rejected(self):
        cards = [types.SimpleNamespace(accelerator=types.SimpleNamespace(value="A100"))]
        self.assertIsNotNone(guard.stop_reason(self.job, self.current, cards, 1001))

    def test_g4_job_uses_its_own_rate_and_hardware_policy(self):
        self.job["policy"] = dict(guard.POLICY, gpu="G4", max_rate_cu_per_hour=12, max_cu=0.25)
        self.current["rate_cu_per_hour"] = 6.7
        cards = [types.SimpleNamespace(accelerator=types.SimpleNamespace(value="G4"))]
        self.assertIsNone(guard.stop_reason(self.job, self.current, cards, 1010))
        self.assertIsNotNone(guard.stop_reason(self.job, self.current, self.t4, 1010))

    def test_cpu_recovery_uses_default_variant_without_gpu(self):
        from colab_cli.client import Variant, Accelerator, GetAssignmentResponse
        self.job["policy"] = dict(guard.POLICY,gpu="NONE")
        class Api:
            def _get_assignment(inner, notebook_id, variant, accelerator):
                self.assertEqual(variant,Variant.DEFAULT)
                self.assertEqual(accelerator,Accelerator.NONE)
                return GetAssignmentResponse(acc="NONE",nbh="test",token="dummy",variant=Variant.DEFAULT)
        self.assertIsNone(guard.owned_endpoint(Api(),self.job))

    def test_known_preexisting_cpu_does_not_abort_owned_g4_job(self):
        self.job["policy"] = dict(guard.POLICY,gpu="G4",max_cu=1.5,max_rate_cu_per_hour=12)
        self.job["baseline_endpoints"] = ["preexisting-user-cpu"]
        self.current["rate_cu_per_hour"] = 6.8
        cpu=types.SimpleNamespace(endpoint="preexisting-user-cpu",accelerator=types.SimpleNamespace(value="NONE"))
        gpu=types.SimpleNamespace(endpoint="our-g4",accelerator=types.SimpleNamespace(value="G4"))
        self.assertIsNone(guard.stop_reason(self.job,self.current,[cpu,gpu],1010))
        extra=types.SimpleNamespace(endpoint="unexpected-new",accelerator=types.SimpleNamespace(value="G4"))
        self.assertIsNotNone(guard.stop_reason(self.job,self.current,[cpu,gpu,extra],1010))

    def test_cleanup_recovers_lost_cli_state_and_only_stops_owned_uuid(self):
        from colab_cli.client import Assignment, GetAssignmentResponse, RuntimeProxyInfo, Variant
        class Api:
            live = True
            stopped = []
            def _get_assignment(inner, notebook_id, variant, accelerator):
                self.assertEqual(str(notebook_id), self.job["notebook_uuid"])
                if inner.live:
                    return Assignment(endpoint="owned-test", runtimeProxyInfo=RuntimeProxyInfo(
                        token="dummy-test-token", tokenExpiresInSeconds=60, url="https://example.invalid"))
                return GetAssignmentResponse(acc="T4", nbh="test", token="dummy", variant=Variant.GPU)
            def unassign(inner, endpoint):
                inner.stopped.append(endpoint)
                inner.live = False
        api = Api()
        self.assertTrue(guard.cleanup(api, self.job))
        self.assertEqual(api.stopped, ["owned-test"])

    def test_network_failure_never_reports_successful_cleanup(self):
        with patch.object(guard, "owned_endpoint", side_effect=ConnectionError), patch.object(guard.time, "sleep"):
            self.assertFalse(guard.cleanup(object(), self.job))

    def test_absence_after_timeout_is_not_confirmed_before_settlement(self):
        self.job['name']='test-uncertain'
        with tempfile.TemporaryDirectory() as directory,patch.object(guard,'PRIVATE',Path(directory)),patch.object(guard.time,'time',return_value=1010),patch.object(guard,'owned_endpoint',return_value=None):
            guard.mark_allocation_uncertain(self.job)
            self.assertFalse(guard.cleanup(object(),self.job))

    def test_late_assignment_is_released_using_original_uuid(self):
        self.job['name']='test-late'
        api=types.SimpleNamespace(unassign=unittest.mock.Mock())
        with tempfile.TemporaryDirectory() as directory,patch.object(guard,'PRIVATE',Path(directory)),patch.object(guard.time,'time',return_value=1010),patch.object(guard,'owned_endpoint',side_effect=['late-owned-g4',None]):
            guard.mark_allocation_uncertain(self.job)
            self.assertTrue(guard.cleanup(api,self.job))
            api.unassign.assert_called_once_with('late-owned-g4')

    def test_controller_done_claim_is_independently_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'job.json'
            guard.write_json(path,self.job)
            guard.write_json(path.with_suffix('.done'),{'cleanup_verified':True})
            with patch.object(guard,'client'),patch.object(guard,'cleanup',return_value=True) as cleanup:
                guard.watchdog(path)
            cleanup.assert_called_once()
            self.assertIn('independent watchdog',path.with_suffix('.release-audit.json').read_text())

    def test_transient_monitor_error_latches_release_until_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'job.json'
            guard.write_json(path,self.job)
            path.with_suffix('.pid').write_text('1')
            with patch.object(guard,'client'),patch.object(guard,'usage',side_effect=[ConnectionError,self.current,RuntimeError]) as usage,patch.object(guard,'cleanup',side_effect=[False,True]),patch.object(guard.time,'time',return_value=1010),patch.object(guard.time,'sleep'):
                guard.watchdog(path)
            self.assertEqual(usage.call_count,2)
            self.assertTrue(path.with_suffix('.release-audit.json').exists())

if __name__ == "__main__":
    unittest.main()
