import torchtext

from collections import Counter
from typing import Generator, List, Tuple, Union, Iterable

from . import SourceField, TargetField
from .. import src_field_name, tgt_field_name, index_field_name, src_idx_field_name


def _read_corpus(path: str) -> Generator[List[str], None, None]:
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            yield line


def _read_examples(path: str, delimiter: str = ',') -> Generator[List[str], None, None]:
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            yield line.split(delimiter)


class Seq2SeqExample(torchtext.data.Example):

    def get_src(self):
        return getattr(self, src_field_name)

    def get_tgt(self):
        return getattr(self, tgt_field_name)

    def __eq__(self, other):
        return isinstance(other, torchtext.data.Example) and \
               vars(self).keys() == vars(other).keys() and \
               all(getattr(self, attr) == getattr(other, attr) for attr in vars(self))

    def __hash__(self):
        return hash(sum(hash(' '.join(getattr(self, attr))) for attr in vars(self)))


class Seq2SeqDataset(torchtext.data.Dataset):
    r"""
    Represents a set of :class:`torchtext.data.Example` objects composed of source and (optionally) target sequences.

    Args:
        examples (list[tuple]): list of data examples
        src_field (torchtext.data.Field): source sequence field
        tgt_field (torchtext.data.Field, optional): target sequence field (default: None)
        dynamic (bool, optional): if True, builds a "dynamic vocabulary", or :class:`torchtext.vocab.Vocab` for each
            source sequence, allowing the target to copy OOV inputs from the source. The idea of a dynamic vocabulary
            is taken from [OpenNMT's dynamic_dict](https://github.com/OpenNMT/OpenNMT-py/blob/95aeefb4377617b94f94090d375eb8ae03dbd6c4/onmt/opts.py#L303).
            (default: True)
        **kwargs: additional arguments to pass to :class:`torchtext.data.Dataset` init
    """

    def __init__(self,
                 examples: Iterable[Union[List, Tuple]],
                 src_field: torchtext.data.Field,
                 tgt_field: torchtext.data.Field = None,
                 dynamic: bool = True,
                 **kwargs):
        self.src_field = src_field
        self.tgt_field = tgt_field

        self.dynamic = dynamic
        self.dynamic_vocab = []

        self._init_fields()

        super(Seq2SeqDataset, self).__init__(self.construct_seq2seq_examples(examples),
                                             self.fields,
                                             **kwargs)

    def _init_fields(self) -> None:
        self.fields = [(index_field_name, torchtext.data.Field(sequential=False,
                                                               use_vocab=False)),
                       (src_field_name, self.src_field)]

        if self.tgt_field is not None:
            self.fields.append((tgt_field_name, self.tgt_field))

        if self.dynamic:
            src_index_field = torchtext.data.Field(sequential=True,
                                                   use_vocab=False,
                                                   pad_token=0,
                                                   batch_first=True)
            self.fields.append((src_idx_field_name, src_index_field))

    def construct_seq2seq_examples(self, examples: Iterable[Union[List, Tuple]]) -> List[Seq2SeqExample]:
        if self.dynamic:
            examples = self._add_dynamic_vocab(examples)

        return [Seq2SeqExample.fromlist([i] + list(data), self.fields) for i, data in enumerate(examples)]

    def _add_dynamic_vocab(self, examples: Iterable[Union[List, Tuple]]) -> Generator[List, None, None]:
        tokenize = self.fields[0][1].tokenize  # Tokenize function of the source field
        for example in examples:
            src_seq = tokenize(example[0]) if isinstance(example[0], str) else example[0]
            dy_vocab = torchtext.vocab.Vocab(Counter(src_seq), specials=[])
            src_indices = [dy_vocab.stoi[w] for w in src_seq]

            self.dynamic_vocab.append(dy_vocab)
            yield list(example) + [src_indices]

    @staticmethod
    def from_file(src_path: str,
                  tgt_path: str = None,
                  share_fields_from: torchtext.data.Dataset = None,
                  **kwargs):
        src_list = _read_corpus(src_path)
        tgt_list = _read_corpus(tgt_path) if tgt_path is not None else None

        return Seq2SeqDataset.from_list(src_list, tgt_list, share_fields_from, **kwargs)

    @staticmethod
    def from_example_file(data_path: str,
                          delimiter: str = '\t',
                          share_fields_from: torchtext.data.Dataset = None,
                          **kwargs):
        examples = _read_examples(data_path, delimiter=delimiter)
        src_field = SourceField() if share_fields_from is None else share_fields_from.fields[src_field_name]
        tgt_field = TargetField() if share_fields_from is None else share_fields_from.fields[tgt_field_name]

        return Seq2SeqDataset(examples, src_field, tgt_field, **kwargs)

    @staticmethod
    def from_list(src_list: Iterable[Union[str, List, Tuple]],
                  tgt_list: Iterable[Union[str, List, Tuple]] = None,
                  share_fields_from: torchtext.data.Dataset = None,
                  **kwargs):
        src_field = SourceField() if share_fields_from is None else share_fields_from.fields[src_field_name]
        tgt_field = None
        if tgt_list is not None:
            tgt_field = TargetField() if share_fields_from is None else share_fields_from.fields[tgt_field_name]
            return Seq2SeqDataset(zip(src_list, tgt_list), src_field, tgt_field, **kwargs)

        return Seq2SeqDataset(zip(src_list), src_field, tgt_field, **kwargs)

    def build_vocab(self, src_vocab_size, tgt_vocab_size):
        self.src_field.build_vocab(self, max_size=src_vocab_size)
        if self.tgt_field:
            self.tgt_field.build_vocab(self, max_size=tgt_vocab_size)
