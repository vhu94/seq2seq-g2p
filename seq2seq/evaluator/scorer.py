import logging
from enum import Enum

import edit_distance

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Tuple, Iterator
from tqdm import tqdm

from seq2seq.data import Seq2SeqDataset

logger = logging.getLogger(__name__)


class Task(Enum):
    G2P = 1


class Scorer(ABC):

    @abstractmethod
    def do_scoring(self, reference_data: Seq2SeqDataset, predictions: Iterator[Tuple[int, List]]) -> Dict[str, Any]:
        r"""
        Score model predictions based on task-specific evaluation metrics

        Args:
            reference_data (Seq2SeqDataset): dataset containing reference examples
            predictions (Iterator[Tuple[int, List]]): model predictions --
                see :func:`seq2seq.evaluator.Predictor.batch_predict`

        Returns:
            dictionary containing task-specific evaluation metrics
        """
        pass


class G2pScorer(Scorer):

    def do_scoring(self, reference_data: Seq2SeqDataset, predictions: Iterator[Tuple[int, List]]) -> Dict[str, Any]:
        errors = {}
        for idx, predicted in tqdm(predictions):
            word = ' '.join(reference_data[idx].get_src())
            reference = reference_data[idx].get_tgt()[1:-1]    # remove <sos> and <eos>

            num_errors, error_codes = self._phone_error(reference, predicted)
            if word not in errors or num_errors < errors[word][0]:
                errors[word] = (num_errors, error_codes, len(reference))

            if num_errors != 0:
                logger.debug(f'{word} : [{" ".join(reference)}]\t[{" ".join(predicted)}]\n')

        word_errors = sum(1 for err in errors.values() if err[0] > 0)
        symbol_errors, type_error, total_symbols = tuple(sum(err, err[0])
                                                         for err in zip((0, Counter(), 0), *errors.values()))

        percent_string_errors = word_errors / len(errors)
        percent_symbol_errors = symbol_errors / total_symbols

        logger.debug(f'Symbol errors: {symbol_errors}, Type errors: {type_error}, '
                     f'Total symbols: {total_symbols}, Word errors: {word_errors}')

        logger.info(f'[PER]: {percent_symbol_errors}')
        logger.info(f'[WER]: {percent_string_errors}')

        return {'PER': percent_symbol_errors, 'WER': percent_string_errors}

    def _phone_error(self, reference: List[Any], prediction: List[Any]) -> Tuple[int, Counter]:
        r"""
        Compute the minimum edit distance, or Levenshtein distance between two sequences

        Args:
            reference (List[Any]): reference sequence
            prediction (List[Any]): predicted sequence

        Returns:
            the edit distance and the number of insertions/substitutions/deletions required
        """
        matcher = edit_distance.SequenceMatcher(reference, prediction,
                                                action_function=edit_distance.highest_match_action)
        errors = matcher.distance()
        op_codes = matcher.get_opcodes()
        error_codes = Counter([op[0] for op in op_codes])

        return errors, error_codes


def get_scorer(task: str) -> Scorer:
    if task.upper() == Task.G2P.name:
        return G2pScorer()
    else:
        raise NotImplemented(f'Scoring for {task} has not been implemented.')
