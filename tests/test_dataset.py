import os
import unittest

import torchtext

import seq2seq

from pathlib import Path
from parameterized import parameterized
from typing import Iterable, Tuple, List, Union

from seq2seq.data import Seq2SeqDataset, SourceField, TargetField


class TestDataset(unittest.TestCase):
    data_path = Path().absolute() / 'data'

    src_path = data_path / 'src.txt'
    tgt_path = data_path / 'tgt.txt'
    examples_path = data_path / 'examples.tsv'

    src_list = [['1', '2', '3'], ['4', '5', '6', '7']]
    tgt_list = [['1', '2', '3'], ['4', '5', '6', '7']]

    @classmethod
    def setUpClass(cls):
        cls.temp_file = 'temp'

        cls.src_only_dataset = Seq2SeqDataset(zip(cls.src_list), SourceField(), dynamic=False)
        cls.src_tgt_dataset = Seq2SeqDataset(zip(cls.src_list, cls.tgt_list), SourceField(), TargetField(), dynamic=False)

    @parameterized.expand([
        ('source_only', src_path, None),
        ('source_and_target', src_path, tgt_path)
    ])
    def test_indices(self, name: str, src_file: str, tgt_file: str):
        dataset = Seq2SeqDataset.from_file(src_file, tgt_file, dynamic=False)
        dataset.build_vocab(1000, 1000)
        batch_size = 25

        generator = torchtext.data.BucketIterator(dataset, batch_size, device=-1)
        self.assertTrue(all(hasattr(batch, seq2seq.index_field_name) for batch in generator),
                        'Every batch should have an index field')
        self.assertTrue(all(i == ex.index for i, ex in enumerate(dataset.examples)),
                        'Example index should match the example list indices')

    @parameterized.expand([
        ('source_only', src_path, None, 2, 100),
        ('source_and_target', src_path, tgt_path, 3, 100)
    ])
    def test_init_from_file(self, name: str, src_file: str, tgt_file: str, num_fields: int, dataset_size: int):
        dataset = Seq2SeqDataset.from_file(src_file, tgt_file, dynamic=False)

        self.assertEqual(num_fields, len(dataset.fields))
        self.assertEqual(dataset_size, len(dataset))
        self.assertTrue(all(hasattr(ex, field) for field in dataset.fields for ex in dataset.examples))

    @parameterized.expand([
        ('source_only', src_list, None, 'src_only_dataset'),
        ('source_and_target', src_list, tgt_list, 'src_tgt_dataset')
    ])
    def test_init_from_list(self, name: str,
                            src_list: Iterable[Union[str, List, Tuple]],
                            tgt_list: Iterable[Union[str, List, Tuple]],
                            ref_name: str):
        dataset = Seq2SeqDataset.from_list(src_list=src_list, tgt_list=tgt_list, dynamic=False)
        ref_dataset = getattr(self, ref_name)

        self.assertEqual(len(ref_dataset), len(dataset),
                         'Number of examples in the reference and list initialized dataset should be equal')
        self.assertEqual(ref_dataset.examples, dataset.examples,
                         'Examples from list initialized dataset should be the same as reference examples')

    @parameterized.expand([
        ('source_and_target', zip(src_list, tgt_list), 'src_tgt_dataset')
    ])
    def test_init_from_example_file(self, name: str, corpus: Iterable[Tuple], ref_name: str):
        # create example file
        with open(self.temp_file, 'w') as tmp:
            for entry in corpus:
                tmp.write('\t'.join(' '.join(item) for item in entry) + '\n')
        self.addCleanup(os.remove, self.temp_file)

        from_examples = Seq2SeqDataset.from_example_file(self.temp_file, dynamic=False)
        ref_dataset = getattr(self, ref_name)

        self.assertEqual(len(ref_dataset), len(from_examples),
                         'Number of examples in the reference and example file initialized dataset should be equal')
        self.assertEqual(ref_dataset.examples, from_examples.examples,
                         'Examples from example file initialized dataset should be the same as reference examples')

    @parameterized.expand([
        ('from_file_source_only', Seq2SeqDataset.from_file(src_path, dynamic=True)),
        ('from_file_source_and_target', Seq2SeqDataset.from_file(src_path, tgt_path, dynamic=True)),
        ('from_list_source_only', Seq2SeqDataset.from_list(src_list, None, dynamic=True)),
        ('from_list_source_and_target', Seq2SeqDataset.from_list(src_list, tgt_list, dynamic=True)),
        ('from_example_file_source_and_target', Seq2SeqDataset.from_example_file(examples_path, dynamic=True))
    ])
    def test_dynamic(self, name: str, dataset: Seq2SeqDataset):
        self.assertTrue(seq2seq.src_idx_field_name in dataset.fields)
        for i, ex in enumerate(dataset.examples):
            self.assertEqual(i, ex.index, 'Example index should match the example list indices')
            src_vocab = dataset.dynamic_vocab[i]
            for tok, tok_id in zip(ex.src, ex.src_index):
                self.assertEqual(src_vocab.stoi[tok], tok_id, 'Source sequence index matches dynamic vocabulary index')
