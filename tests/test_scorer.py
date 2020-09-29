import unittest
from typing import List, Tuple, Iterator

from parameterized import parameterized

from seq2seq.data import Seq2SeqDataset
from seq2seq.evaluator.scorer import get_scorer


class TestScorer(unittest.TestCase):

    @parameterized.expand([
        ('all_correct',
         ['n a n n y', 'g h o s t'], ['n {" n i', 'g oU" s t'],
         [(0, ['n', '{"', 'n', 'i']), (1, ['g', 'oU"', 's', 't'])],
         0.0, 0.0),
        ('mispronunciations',
         ['n a n n y', 'g h o s t'], ['n {" n i', 'g oU" s t'],
         [(0, ['n', 'n', 'i']), (1, ['g', '{"', 's', 't', 's'])],
         3 / 8, 1.0),
        ('smallest_edit_distance',
         ['n a n n y', 's n o o k', 's n o o k'], ['n {" n i', 's n u" k', 's n U" k'],
         [(0, ['n', '{"', 'n', 'i']), (1, ['s', 'u"', 'k']), (2, ['s', 'u"', 'k'])],
         1 / 8, 1 / 2),
        ('correct_alternate_pronunciation',
         ['n a n n y', 's n o o k', 's n o o k'], ['n {" n i', 's n u" k', 's n U" k'],
         [(0, ['n', '{"', 'n', 'e']), (1, ['s', 'n', 'u"', 'k']), (2, ['s', 'n', 'u"', 'k'])],
         1 / 8, 1 / 2)
    ])
    def test_g2p_scorer(self, name: str, src_list: List, tgt_list: List,
                        predictions: Iterator[Tuple[int, List]],
                        per: float, wer: float):
        scorer = get_scorer('g2p')
        scores = scorer.do_scoring(Seq2SeqDataset.from_list(src_list, tgt_list), predictions)

        self.assertEqual(per, scores['PER'])
        self.assertEqual(wer, scores['WER'])
