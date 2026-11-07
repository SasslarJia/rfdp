"""Configuration contracts, not retrieval experiments."""
import copy
import json
from pathlib import Path
import unittest
from experiments.continuous_fitting import research_config as cfg
from Test_RFDP.test_f1_config import checked_base


class ResearchConfigTests(unittest.TestCase):
    def test_eight_main_and_four_disabled_reserve_recipes(self):
        root=Path(__file__).resolve().parents[2]/'configs'
        main=list((root/'converged-f1').glob('*.json'))
        backup=list((root/'diagonal-free').glob('*.json'))
        self.assertEqual((len(main),len(backup)),(8,4))
        for file in main+backup:
            record=json.loads(file.read_text())
            self.assertEqual(record,cfg.recipe(record['input_id'],record['source']))
            self.assertEqual(record['enabled'],file in main)
            self.assertNotEqual(record['source'],'original')
            with self.assertRaises(ValueError):cfg.validate(record,require_bound=True)

    def test_binding_preserves_inputs_but_not_old_algorithm(self):
        for name in cfg.legacy.FROZEN_INPUTS:
            base=checked_base(name)
            for source in cfg.SOURCES:
                bound=cfg.bind(cfg.recipe(name,source),base,output_root='/new/research',commit='d'*40)
                self.assertEqual(bound['input_sha256'],base['input_sha256'])
                self.assertEqual(bound['commit'],'d'*40)
                self.assertTrue(bound['output'].endswith('/'+source))
                self.assertNotIn('f1_updates',bound)
                cfg.validate(bound,require_bound=True)
                if source==cfg.BACKUP:
                    with self.assertRaisesRegex(ValueError,'disabled'):
                        cfg.validate(bound,require_bound=True,require_enabled=True)
                    enabled=cfg.bind(cfg.recipe(name,source),base,output_root='/new/research',commit='d'*40,enable_backup=True)
                    cfg.validate(enabled,require_bound=True,require_enabled=True)

    def test_no_legacy_fields_or_wrong_fixed_settings(self):
        record=cfg.recipe('market-osnet',cfg.MAIN[0])
        for key,value in [('nu',2),('score_tolerance',1e-4),('f1_updates',1),('depth',50),('retain_initial_state',False)]:
            bad=copy.deepcopy(record);bad[key]=value
            with self.assertRaises(ValueError):cfg.validate(bad)
        self.assertEqual(cfg.validate({'source':'original'},require_bound=True),{})
        with self.assertRaises(ValueError):cfg.validate({'source':cfg.MAIN[0]})


if __name__=='__main__':unittest.main()
