import sys
import unittest

import numpy as np
import rdkit.Chem as Chem

from askcos.retrosynthetic.mcts.tree_builder import MCTS


class DummyHistorian:
    def lookup_smiles(self, *args, **kwargs):
        return {'as_reactant': 0, 'as_product': 0}


# Check if any other test modules have been imported
# Hacky workaround since tensorflow is not fork-safe
# This test module must be run alone in it's own process to work
test_modules = [mod for mod in sys.modules if mod.endswith('_test') and not mod.endswith('tree_builder_test')]


@unittest.skipIf(test_modules, 'Tree builder test must be run in independent process.')
class TestMCTSTreeBuilder(unittest.TestCase):
    """Contains functional tests for the MCTS class."""

    @classmethod
    def setUpClass(cls):
        cls.nproc = 2
        cls.mcts = MCTS(nproc=cls.nproc, chemhistorian=DummyHistorian())

    def assertDictAlmostEqual(self, first, second, rtol=1e-7, atol=0, msg=None, ignore_keys=None):
        """
        Recursive dictionary comparison, wth the following properties:
        - dict keys must be exactly equal, order is ignored
        - numerical dict values are compared using np.allclose
        - non-numerical dict values must be exactly equal
        - lists are assumed to be homogeneous, and the contents are compared
        """
        self.assertEqual(set(first.keys()), set(second.keys()))

        for key, value in first.items():
            if ignore_keys and key in ignore_keys:
                continue
            if isinstance(value, dict):
                self.assertDictAlmostEqual(value, second[key], rtol=rtol, atol=atol, msg=msg, ignore_keys=ignore_keys)
            elif isinstance(value, (str, bool)):
                self.assertEqual(value, second[key], msg=msg)
            elif isinstance(value, (int, float, np.ndarray, np.generic)):
                np.testing.assert_allclose(value, second[key], rtol=rtol, atol=atol, err_msg=msg)
            elif isinstance(value, list):
                self.assertListAlmostEqual(value, second[key], rtol=rtol, atol=atol, msg=msg, ignore_keys=ignore_keys)
            else:
                raise NotImplementedError(type(value))

    def assertListAlmostEqual(self, first, second, rtol=1e-7, atol=0, msg=None, ignore_keys=None):
        """
        Recursive list comparison, with the following properties:
        - contents are assumed to be of homogeneous type
        - numerical values are compared using np.allclose
        - non-numerical values must be exactly equal
        """
        self.assertEqual(len(first), len(second))

        if len(first) == 0:
            # Empty list
            return

        if isinstance(first[0], dict):
            for a, b in zip(first, second):
                self.assertDictAlmostEqual(a, b, rtol=rtol, atol=atol, msg=msg, ignore_keys=ignore_keys)
        elif isinstance(first[0], (str, bool)):
            self.assertEqual(first, second, msg=msg)
        elif isinstance(first[0], (int, float, np.ndarray, np.generic)):
            np.testing.assert_allclose(first, second, rtol=rtol, atol=atol, err_msg=msg)
        elif isinstance(first[0], list):
            for a, b in zip(first, second):
                self.assertListAlmostEqual(a, b, rtol=rtol, atol=atol, msg=msg, ignore_keys=ignore_keys)
        else:
            raise NotImplementedError(type(first[0]))

    def check_path(self, path, target):
        """Check that a path result has some expected features."""
        self.assertIsInstance(path, dict)
        chem_keys = {'id', 'ppg', 'children', 'smiles', 'is_chemical', 'as_product', 'as_reactant', 'attributes', 'terminal'}
        self.assertEqual(chem_keys, set(path.keys()))
        self.assertEqual(path['smiles'], target)
        self.assertTrue(path['is_chemical'])

        rxn = path['children'][0]
        rxn_keys = {'id', 'tforms', 'template_set', 'template_score', 'necessary_reagent', 'children', 'num_examples', 'plausibility',
                    'is_reaction', 'smiles', 'rank', 'rms_molwt', 'num_rings', 'precursor_smiles'}
        self.assertEqual(rxn_keys, set(rxn.keys()))
        self.assertIn(target, path['smiles'])
        self.assertTrue(rxn['is_reaction'])

    def test_diphenhydramine(self):
        """Test that MCTS works for diphenhydramine."""
        self.assertTrue(self.mcts)

        smiles = 'CN(C)CCOC(c1ccccc1)c1ccccc1'
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
        status, paths, graph = self.mcts.get_buyable_paths(
            smiles,
            nproc=self.nproc,
            expansion_time=30,
            max_cum_template_prob=0.995,
            template_count=100,
            termination_logic={'or': ['buyable']},
            soft_reset=True,  # do not kill workers since there are more tests
            soft_stop=True,  # do not kill workers since there are more tests
        )

        # Check that the number of chemical and reactions is in a reasonable range
        # Use very conservative lower bound to account for non-deterministic behavior
        self.assertGreater(status[0], 40)
        self.assertGreater(status[1], 100)
        self.assertGreater(len(paths), 100)

        # Check that the top path result has the expected format
        self.check_path(paths[0], smiles)

    def test_scopolamine(self):
        """Test that MCTS works for scopolamine."""
        self.assertTrue(self.mcts)

        smiles = 'Cc1ncc([N+](=O)[O-])n1CC(C)O'
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
        status, paths, graph = self.mcts.get_buyable_paths(
            smiles,
            nproc=self.nproc,
            expansion_time=30,
            max_cum_template_prob=0.995,
            template_count=100,
            termination_logic={'or': ['buyable']},
            soft_reset=True,  # do not kill workers since there are more tests
            soft_stop=True,  # do not kill workers since there are more tests
        )

        # Check that the number of chemical and reactions is in a reasonable range
        # Use very conservative lower bound to account for non-deterministic behavior
        self.assertGreater(status[0], 60)
        self.assertGreater(status[1], 80)
        self.assertGreater(len(paths), 10)

        # Check that the top path result has the expected format
        self.check_path(paths[0], smiles)

    def test_random(self):
        """Test that MCTS works for a random compound."""
        self.assertTrue(self.mcts)

        smiles = 'CCCCCN(CCCCC)CCCC(=O)OCCC'
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
        status, paths, graph = self.mcts.get_buyable_paths(
            smiles,
            nproc=self.nproc,
            expansion_time=30,
            max_cum_template_prob=0.995,
            template_count=100,
            termination_logic={'or': ['buyable']},
            soft_reset=True,  # do not kill workers since there are more tests
            soft_stop=True,  # do not kill workers since there are more tests
        )

        # Check that the number of chemical and reactions is in a reasonable range
        # Use very conservative lower bound to account for non-deterministic behavior
        self.assertGreater(status[0], 80)
        self.assertGreater(status[1], 100)
        self.assertGreater(len(paths), 400)

        # Check that the top path result has the expected format
        self.check_path(paths[0], smiles)

    @classmethod
    def tearDownClass(cls):
        for p in cls.mcts.workers:
            p.terminate()


def test_psilocybin(self):
    """Test that MCTS works for psilocybin."""
    self.assertTrue(self.mcts)

    smiles = 'CN(C)CCC1=CNC2=C1C(=CC=C2)OP(=O)(O)O'
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
    status_result, paths_result, graph = self.mcts.get_buyable_paths(
        smiles,
        nproc=self.nproc,
        expansion_time=30,
        max_cum_template_prob=0.995,
        template_count=100,
        soft_reset=True,  # do not kill workers since there are more tests
        soft_stop=True,  # do not kill workers since there are more tests
    )

    # Check that the number of chemical and reactions is in a reasonable range
    # Use very conservative lower bound to account for non-deterministic behavior
    self.assertGreater(status_result[0], 40)
    self.assertGreater(status_result[1], 200)

    # Check that the top path result matches the expected one
    output_path_1 = paths_result[0]
    expected_path_1 = {
        'as_reactant': 59,
        'children': [
            {'id': 87,
             'tforms': ['5e1f4b6e6348832850997243'],
             'template_set': ['reaxys'],
             'template_score': 0.07315371,
             'necessary_reagent': '',
             'children': [
                 {'as_reactant': 2438,
                  'children': [],
                  'ppg': 1.0,
                  'is_chemical': True,
                  'id': 79,
                  'as_product': 102,
                  'smiles': 'BrC(c1ccccc1)c1ccccc1'},
                 {'as_reactant': 3643,
                  'children': [],
                  'ppg': 1.0,
                  'is_chemical': True,
                  'id': 84,
                  'as_product': 383,
                  'smiles': 'CN(C)CCO'}
             ],
             'num_examples': 185,
             'plausibility': 0.9988686442375183,
             'is_reaction': True,
             'smiles': 'BrC(c1ccccc1)c1ccccc1.CN(C)CCO>>CN(C)CCOC(c1ccccc1)c1ccccc1'
             }
        ],
        'ppg': 1.0,
        'is_chemical': True,
        'id': 4,
        'as_product': 44,
        'smiles': 'CN(C)CCOC(c1ccccc1)c1ccccc1'
    }

    # The 'id' field is non-deterministic, so ignore it
    self.assertDictAlmostEqual(output_path_1, expected_path_1, rtol=1e-3,
                               ignore_keys=[
                                   'id', 'as_reactant', 'as_product'],
                               msg='Obtained different output: {}'.format(output_path_1))


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
