import hashlib
import os
from pymongo import MongoClient
from askcos.utilities.io.logger import MyLogger
import rdkit.Chem as Chem
import gzip
import json
import askcos.global_config as gc
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(4)
historian_loc = 'chemhistorian'


class ChemHistorian:
    """Looks up chemicals to see how often they have been seen in Reaxys.

    Attributes:
        CHEMICALS_DB (MongoDB): Database of chemicals.
        use_db (bool): Flag to use mongo database
        occurrences (dict): Dictionary of smiles -> occurrence frequencies
    """

    def __init__(self, CHEMICALS_DB=None, use_db=False):
        """Initializes ChemHistorian.

        Args:
            CHEMICALS (None or MongoDB): ?? (default: {None})
        """
        self.use_db = use_db
        self.CHEMICALS_DB = CHEMICALS_DB

        self.occurrences = {}

    def load_databases(self):
        """Loads the history data from the mongo database."""
        db_client = MongoClient(gc.MONGO['path'], gc.MONGO['id'], connect=gc.MONGO['connect'])
        db = db_client[gc.CHEMICALS['database']]
        self.CHEMICALS_DB = db[gc.CHEMICALS['collection']]

    def dump_to_file(self, file_path=gc.CHEMICALS['file_name']):
        """Writes the data from the mongo datbases to a local file.

        Args:
            file_path (str, optional): Path to the output file.
                (default: {gc.historian_data})
        """

        chemicals = []
        for k, v in self.occurrences.items():
            tmp = v.copy()
            tmp['smiles'] = k
            chemicals.append(tmp)

        with gzip.open(file_path, 'wb') as f:
            json.dump(chemicals, f)
        MyLogger.print_and_log("Saved to {}".format(file_path), historian_loc)

    def load_from_file(self, file_path=gc.CHEMICALS['file_name']):
        """Loads the data for the pricer from a locally stored file.

        Args:
            file_path (str, optional): Path to the input file.
                (default: {gc.historian_data})
            refs (bool, optional): Whether to include the references or just
                the counts. (default: {False})
            compressed (bool, optional): Whether the data is compressed.
                (default: {False})

        Raises:
            ValueError: If file does not exist.
        """

        if not os.path.isfile(file_path):
            raise ValueError('File does not exist!')

        MyLogger.print_and_log(
            'Loading chemhistorian from file...', historian_loc)

        with gzip.open(file_path, 'rb') as f:
            chemicals = json.loads(f.read().decode('utf-8'))

        for chem in chemicals:
            smiles = chem.pop('smiles')
            self.occurrences[smiles] = chem

    def load(self):
        """Loads the object from a MongoDB collection.

        Args:
            CHEMICAL_DB (None or ??, optional): ??
            refs (bool, optional): Whether to include references.
                (default: {True})
        """
        MyLogger.print_and_log(
            'Loading historian.', historian_loc)

        if not self.use_db:
            self.load_from_file()
        elif not self.CHEMICALS_DB:
            self.load_databases()

        MyLogger.print_and_log('Historian is fully loaded.', historian_loc)

    def lookup_smiles(self, smiles, alreadyCanonical=False, isomericSmiles=True, template_sets=['reaxys']):
        """Looks up number of occurances by SMILES.

        Tries it as-entered and then re-canonicalizes it in RDKit unless the
        user specifies that the string is definitely already canonical.

        Args:
            smiles (str): SMILES string of molecule to look up.
            alreadyCanonical (bool, optional): Whether to SMILES string has
                already been canonicalized. (default: {False})
            isomericSmiles (bool, optional): Whether the SMILES string is
                isomeric. Must be true, since function is only intended for
                isomeric. (default: {True})
            refs (bool, optional): Whether to include references.
                (default: {False})

        Returns:
            dict: Labeled dictionaty from tup_to_dict().

        Raises:
            ValueError: If isomericSmiles is False.
        """

        default_result = {
            'as_reactant': 0,
            'as_product': 0
        }
        if not isomericSmiles:
            raise ValueError('Not intended to be used for non-isomeric!')

        if not alreadyCanonical:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return default_result
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
        hashed_smiles = str(
            int(hashlib.md5(smiles.encode('utf-8')).hexdigest(), 16))
        if self.use_db:
            docs = [self.CHEMICALS_DB.find_one(
                    {
                    'smiles': {'$in': [smiles, hashed_smiles]},
                    'template_set' : t_s
                    },
                    {'as_reactant': 1, 'as_product': 1}) for t_s in template_sets]

            #aggregate results from different historian DBs
            result = default_result
            for doc in docs:
                if doc:
                    result['as_reactant'] += doc['as_reactant']
                    result['as_product'] += doc['as_product']
            return result
        else:
            result = self.occurrences.get(smiles, None)
            if result is not None:
                return result
            result = self.occurrences.get(hashed_smiles, None)
            if result is not None:
                return result
            return default_result

    def compress_keys(self):
        """Converts keys to hashed values to save space."""
        new_occurrences = {}
        for k in self.occurrences.keys():
            k_compressed = str(
                int(hashlib.md5(k.encode('utf-8')).hexdigest(), 16))
            new_occurrences[k_compressed] = self.occurrences[k]
        del self.occurrences
        self.occurrences = new_occurrences


if __name__ == '__main__':
    chemhistorian = ChemHistorian()
    print('loading no refs, compressed')
    chemhistorian.load_from_file()
    print(chemhistorian.lookup_smiles('CCCCO'))