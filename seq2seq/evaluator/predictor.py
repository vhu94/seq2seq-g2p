import torch
import torchtext

import seq2seq

from torchtext.vocab import Vocab
from typing import List, Tuple, Generator

from seq2seq.data import Seq2SeqDataset, TargetField
from seq2seq.models import Seq2seq


class Predictor(object):

    def __init__(self, model: Seq2seq, src_vocab: Vocab, tgt_vocab: Vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (torchtext.vocab.Vocab): source sequence vocabulary
            tgt_vocab (torchtext.vocab.Vocab): target sequence vocabulary
        """
        self.model = model.cuda() if torch.cuda.is_available() else model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def batch_predict(self, dataset: Seq2SeqDataset, batch_size: int = 32) -> Generator[Tuple[int, List], None, None]:
        device = 'cuda' if torch.cuda.is_available() else None
        batch_iterator = torchtext.data.BucketIterator(
            dataset=dataset, batch_size=batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)

        model_tgt_eos = self.tgt_vocab.stoi[TargetField.SYM_EOS]
        with torch.no_grad():
            for batch in batch_iterator:
                _, _, other = self.model(batch)
                indexes = getattr(batch, seq2seq.index_field_name)
                for i in range(batch.batch_size):
                    decoded_id_seq = [other['sequence'][step][i].data[0] for step in range(other['length'][i])]
                    decoded_seq = [self.tgt_vocab.itos[tok] for tok in decoded_id_seq if tok != model_tgt_eos]

                    yield indexes[i].item(), decoded_seq

    def predict(self, src_seq: List) -> List:
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (List): list of input tokens in source language

        Returns:
            tgt_seq (List): list of output tokens in target language as predicted
            by the pre-trained model
        """
        with torch.no_grad():
            src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
            if torch.cuda.is_available():
                src_id_seq = src_id_seq.cuda()

            dataset = Seq2SeqDataset.from_list(' '.join(src_seq))
            dataset.vocab = self.src_vocab
            batch = torchtext.data.Batch.fromvars(dataset, 1,
                                                  src=(src_id_seq, [len(src_seq)]), tgt=None)

            _, _, other = self.model(batch)

        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]

        return tgt_seq
