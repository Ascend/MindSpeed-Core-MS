from functools import wraps


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def parse_args_decorator(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'
    parser = _add_network_size_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_training_args(parser)
    parser = _add_data_args(parser)

    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')
    group.add_argument("--use-fused-rmsnorm", action='store_true',
                       help="Use fused rmsnorm.")
    group.add_argument("--use-fused-swiglu", action='store_true',
                       help="Use fused swiglu.")
    group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                       help="Use fused rotary-pos-emb.")
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'Llama2Tokenizer',
                                'PretrainedFromHF',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--local-rank', type=int, default=None,
                       help='Local rank passed from distributed launcher for torch2.x.')
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    # gradient_accumulation_fusion保持常闭
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false', default=False,
                       help='Disable fusing gradient accumulation to weight '
                            'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--pre-tockens', type=int, default=65536,
                       help='pre-tockens is used by Flash attention')
    group.add_argument('--next-tockens', type=int, default=0,
                       help='next-tockens is used by Flash attention')
    group.add_argument('--shape-order', type=str, default='SBH',
                       choices=['SBH', 'BSH', 'BSND'],
                       help='input shape order used by Flash attention')
    return parser


def validate_args_decorator(validate_args):
    @wraps(validate_args)
    def wrapper(args, defaults):
        validate_args(args, defaults)
        if args.use_fused_rmsnorm:
            if args.normalization != "RMSNorm":
                raise AssertionError(
                    '--use-fused-rmsnorm must enable with '
                    '--normalization=RMSNorm, but got normalization'
                    '={}.'.format(args.normalization))
        if args.use_fused_swiglu:
            if not args.swiglu:
                raise AssertionError(
                    '--use-fused-swiglu must enable with --swiglu, '
                    'but --swiglu={}.'.format(args.swiglu))
        if args.use_fused_rotary_pos_emb:
            if not args.use_rotary_position_embeddings:
                raise AssertionError(
                    '--use-fused-rotary-pos-emb must enable with --use-rotary-position-embeddings, '
                    'but --use-rotary-position-embeddings={}.'.format(args.use_rotary_position_embeddings))
    return wrapper
