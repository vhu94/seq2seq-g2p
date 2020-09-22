class Hparams:

    def __init__(self,
                 encoder_hidden_size: int = 128,
                 encoder_rnn_cell: str = 'lstm',
                 n_encoder_layers: int = 2,
                 encoder_embed_dim: int = 128,
                 bidirectional_encoder: bool = True,
                 encoder_dropout: float = 0.25,
                 encoder_input_dropout: float = 0.25,
                 decoder_rnn_cell: str = 'lstm',
                 decoder_dropout: float = 0.25,
                 decoder_input_dropout: float = 0.25,
                 decoder_embed_dim: int = 256,
                 max_sequence_length: int = 50,
                 loss_type: str = 'perplexity',
                 batch_size: int = 64,
                 teacher_forcing_ratio: float = 0.5,
                 input_flag_dim: int = 0,
                 use_attention: bool = True):
        # encoder parameters
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_rnn_cell = encoder_rnn_cell
        self.n_encoder_layers = n_encoder_layers
        self.bidirectional_encoder = bidirectional_encoder
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_input_dropout = encoder_input_dropout
        self.encoder_dropout = encoder_dropout
        self.input_flag_dim = input_flag_dim

        # decoder parameters
        self.decoder_rnn_cell = decoder_rnn_cell
        self.n_decoder_layers = self.n_encoder_layers
        self.decoder_dropout = decoder_dropout
        self.decoder_input_dropout = decoder_input_dropout
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_hidden_size = self.encoder_hidden_size * 2 \
            if self.bidirectional_encoder else self.encoder_hidden_size
        self.use_attention = use_attention

        # data parameters
        self.max_sequence_length = max_sequence_length

        # training parameters
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio


def get_hparams() -> Hparams:
    return Hparams()
