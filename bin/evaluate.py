import argparse
import logging

import torchtext

from pathlib import Path
from typing import Dict, Any

from seq2seq import src_field_name, tgt_field_name
from seq2seq.data import Seq2SeqDataset
from seq2seq.evaluator import Predictor
from seq2seq.evaluator.scorer import get_scorer
from seq2seq.models import Seq2seq
from seq2seq.util.checkpoint import Checkpoint
from utils import setup_logging


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',
                        action='store',
                        dest='test',
                        required=True,
                        help='Path to test data')
    parser.add_argument('--task',
                        action='store',
                        dest='task',
                        default='g2p')
    parser.add_argument('--checkpoint',
                        action='store',
                        dest='checkpoint',
                        required=True,
                        help='Path of the checkpoint to load.')
    parser.add_argument('--log_level',
                        dest='log_level',
                        default='info',
                        help='Logging level')
    parser.add_argument('--logfile',
                        action='store',
                        dest='logfile',
                        help='Path to log file')

    return parser


def load_test_data(test_path: str, src_vocab: torchtext.vocab.Vocab, tgt_vocab: torchtext.vocab.Vocab):
    r"""
    Loads the test dataset and sets the dataset vocab mapping to the original training set input/output vocab

    Args:
        test_path (str): path to test data
        src_vocab (torchtext.vocab.Vocab): source sequence vocabulary
        tgt_vocab (torchtext.vocab.Vocab): target sequence vocabulary

    Returns:
        the test dataset
    """
    test_dataset = Seq2SeqDataset.from_example_file(test_path)

    test_dataset.fields[src_field_name].vocab = src_vocab
    test_dataset.fields[tgt_field_name].vocab = tgt_vocab

    return test_dataset


def evaluate(model: Seq2seq,
             src_vocab: torchtext.vocab.Vocab,
             tgt_vocab: torchtext.vocab.Vocab,
             dataset: Seq2SeqDataset,
             task: str) -> Dict[str, Any]:
    r"""
    Evaluate a trained model on a dataset for some task-specific metrics

    Args:
        model (Seq2seq): trained model
        src_vocab (torchtext.vocab.Vocab): source sequence vocabulary
        tgt_vocab (torchtext.vocab.Vocab): target sequence vocabulary
        dataset (Seq2SeqDataset): test dataset
        task (str): name of task

    Returns:

    """
    predictor = Predictor(model, src_vocab, tgt_vocab)
    predictions = predictor.batch_predict(dataset)
    scorer = get_scorer(task)

    return scorer.do_scoring(test_dataset, predictions)


if __name__ == '__main__':
    args = arg_parser().parse_args()

    setup_logging(args.log_level.upper(), args.logfile)
    logging.info(args)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)

    test_dataset = load_test_data(args.test, checkpoint.input_vocab, checkpoint.output_vocab)
    scores, avg_time, samples = evaluate(model=checkpoint.model,
                                         src_vocab=checkpoint.input_vocab,
                                         tgt_vocab=checkpoint.output_vocab,
                                         dataset=test_dataset,
                                         task=args.task)
