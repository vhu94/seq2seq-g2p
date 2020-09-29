import argparse
import logging

import torch

from pathlib import Path
from typing import Union

from hparams import get_hparams, Hparams
from seq2seq.data import Seq2SeqDataset, TargetField, SourceField
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import (
    EncoderRNN,
    DecoderRNN,
    Seq2seq,
)
from seq2seq.loss import Perplexity, NLLLoss
from utils import setup_logging, set_random


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        action='store',
                        dest='train',
                        help='Path to train data')
    parser.add_argument('--dev',
                        action='store',
                        dest='dev',
                        help='Path to dev data')
    parser.add_argument('--exp_name',
                        action='store',
                        dest='exp_name',
                        default='experiments',
                        help='Path to experiment directory. If load_checkpoint is True, '
                             'then path to checkpoint directory has to be provided')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Maximum number of epochs to run')
    parser.add_argument('--patience',
                        type=int,
                        default=-1,
                        help='The number of epochs to continue training without observing improvement. '
                             'If set to -1, training continues for the maximum number of epochs.')
    parser.add_argument('--checkpoint',
                        action='store',
                        dest='checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--seed',
                        type=int,
                        dest='seed',
                        help='Random seed for initializing RNGs for reproducibility')
    parser.add_argument('--checkpoint_interval',
                        type=int,
                        default=100,
                        dest='checkpoint_interval',
                        help='Number of training steps between checkpoint saves')
    parser.add_argument('--log_level',
                        dest='log_level',
                        default='info',
                        help='Logging level')
    parser.add_argument('--logfile',
                        dest='logfile',
                        help='Path to log file')

    return parser


def init_model(src: SourceField, tgt: TargetField, hparams: Hparams) -> Seq2seq:
    encoder = EncoderRNN(vocab_size=len(src.vocab),
                         max_len=hparams.max_sequence_length,
                         hidden_size=hparams.encoder_hidden_size,
                         bidirectional=hparams.bidirectional_encoder,
                         dropout_p=hparams.encoder_dropout,
                         input_dropout_p=hparams.encoder_input_dropout,
                         rnn_cell=hparams.encoder_rnn_cell,
                         n_layers=hparams.n_encoder_layers,
                         variable_lengths=True)
    decoder = DecoderRNN(vocab_size=len(tgt.vocab),
                         max_len=hparams.max_sequence_length,
                         hidden_size=hparams.decoder_hidden_size,
                         n_layers=hparams.n_decoder_layers,
                         dropout_p=hparams.decoder_dropout,
                         input_dropout_p=hparams.decoder_input_dropout,
                         bidirectional_encoder=hparams.bidirectional_encoder,
                         rnn_cell=hparams.decoder_rnn_cell,
                         eos_id=tgt.eos_id,
                         sos_id=tgt.sos_id,
                         use_attention=hparams.use_attention)
    seq2seq = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    return seq2seq


def prepare_loss(tgt: TargetField, loss_type: str = 'perplexity') -> NLLLoss:
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]

    if loss_type == 'perplexity':
        loss = Perplexity(weight, pad)
    elif loss_type == 'nll':
        loss = NLLLoss(weight, pad)
    else:
        raise NotImplementedError(f'The selected loss type {loss_type} is not implemented.')

    if torch.cuda.is_available():
        loss.cuda()

    return loss


def train(opts: argparse.Namespace, hparams: Hparams, model_dir: Union[str, Path] = 'default') -> Seq2seq:
    set_random(opts.seed)

    # Prepare dataset
    train = Seq2SeqDataset.from_example_file(opts.train)
    train.build_vocab(50000, 50000)
    dev = Seq2SeqDataset.from_example_file(opts.dev, share_fields_from=train.fields) if opts.dev else None
    src = train.src_field
    tgt = train.tgt_field

    # Prepare loss
    loss = prepare_loss(tgt, hparams.loss_type)

    # Load checkpoint if given
    # hparams.input_flag_dim = 1 if train.vowel_aware else 0
    seq2seq = init_model(src, tgt, hparams)

    logging.info(f'Training model with {seq2seq.trainable_parameters()} parameters: {seq2seq}')

    # Train
    t = SupervisedTrainer(experiment_directory=Path(opts.exp_name, model_dir),
                          loss=loss,
                          batch_size=hparams.batch_size,
                          random_seed=opts.seed,
                          checkpoint_every=opts.checkpoint_interval,
                          print_every=opts.checkpoint_interval)
                          # patience=opts.patience)

    model = t.train(seq2seq,
                    train,
                    n_epochs=opts.epochs,
                    resume=opts.checkpoint is not None,
                    dev_data=dev,
                    teacher_forcing_ratio=hparams.teacher_forcing_ratio)
                    # checkpoint_path=opts.checkpoint)

    return model


if __name__ == '__main__':
    # Parse args; get hparams
    args = arg_parser().parse_args()
    hparams = get_hparams()

    # Set up logging
    setup_logging(args.log_level.upper(), args.logfile)
    logging.info(args)

    # Start training
    train(args, hparams)
