import json,tempfile,unittest
from pathlib import Path
import reporting
import run_stage2b as m
class ReportingTests(unittest.TestCase):
    def test_missing_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);reporting.report(p,{'status':'failed','error':'fixture solver failure'})
            r=json.loads((p/'outcome_analysis.json').read_text())
            self.assertEqual(len(r['missing_episodes']),12)
            self.assertEqual(r['comparisons'],[])
            self.assertIn('HOLD',(p/'STAGE2B_RESEARCH_DECISION.md').read_text())
    def test_nonfinite_diagnostic_still_stops(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'nonfinite'):m.Prefix(Path(d)).value('native',[0.],[float('nan')])
    def test_new_worker_still_has_prereset_contract_barrier(self):
        source=Path(m.__file__).read_text()
        self.assertLess(source.index("out/'preregistration_ready.json'"),source.index('env,raw,identity,adapter,audit=environment'))
        self.assertIn("require(reference['identity']==meta['identity']",source)
