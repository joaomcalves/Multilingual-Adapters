# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

from fairseq import utils
from fairseq.models import (
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)


@register_model('multilingual_adapters')
class MultilingualTransformerModelAdapters(FairseqMultiModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
        --freeze-adapters: variable to decide if it is time to train the adapters or the other layers
        --freeze-layers: variable to decide if it is time to freeze the other layers
        --adapter-projection-dim: dimension of the down projection of the adapters
        --lang-adapter: lang-pairs that will use adapter for fine-tunning

    """

    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')
        parser.add_argument('--adapter-projection-dim', type=int, metavar='N',
                            help='dimension of the down projection of the adapters')
        parser.add_argument('--lang-pairs-adapters', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs that will have adapters')
        parser.add_argument('--freeze-adapters', action='store_true',
                            help='variable to decide if it is time to train the adapters or not')
        parser.add_argument('--freeze-layers', action='store_true',
                            help='variable to decide if it is time to freeze all the other layers')
        parser.add_argument('--freeze-embeddings', action='store_true',
                            help='variable to decide if it is time to freeze embeddings')
        parser.add_argument('--adapters-type', default=None)
        parser.add_argument('--languages-adapters', default=None)
        
        

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        multilingual_adapters_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]


        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        freeze_embeddings = args.freeze_embeddings

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)

            if freeze_embeddings:
                freeze_module_params(emb)

            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=src_langs,
                        embed_dim=args.encoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.encoder_embed_path,
                    )
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=tgt_langs,
                        embed_dim=args.decoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.decoder_embed_path,
                    )
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path
                    )
                lang_encoders[lang] = TransformerEncoder(args, task.dicts[lang], encoder_embed_tokens)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim, args.decoder_embed_path
                    )
                lang_decoders[lang] = TransformerDecoder(args, task.dicts[lang], decoder_embed_tokens)
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src)
            decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt)


        return MultilingualTransformerModelAdapters(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True, args=None):
        
        """
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith('models.')
            lang_pair = k.split('.')[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        """
        state_dict_subset = state_dict.copy()
        for model in self.models:
            src_desired_model, tgt_desired_model = model.split('-')
            for k, _ in state_dict.items():
                assert k.startswith('models.')
                k_elements = k.split('.')
                new_k_elements = k_elements.copy()
                src_existent_model, tgt_existent_model = k_elements[1].split('-')
                if k_elements[2] == 'encoder' and src_desired_model == src_existent_model:
                    new_k_elements[1] = model
                    new_key = '.'.join(new_k_elements)
                    if new_key not in state_dict_subset:
                        state_dict_subset[new_key] = state_dict_subset[k].clone()
                if k_elements[2] == 'decoder' and tgt_desired_model == tgt_existent_model:
                    new_k_elements[1] = model
                    new_key = '.'.join(new_k_elements)
                    if new_key not in state_dict_subset:
                        state_dict_subset[new_key] = state_dict_subset[k].clone()
        
        super().load_state_dict(state_dict_subset, strict=False, args=args)

def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False 

@register_model_architecture('multilingual_adapters', 'multilingual_adapters')
def multilingual_adapters_architecture(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', False)
    args.lang_adapter = getattr(args, 'lang__adapter', None)
    args.adapter_projection_dim = getattr(args, 'adapter_projection_dim', 2048)
    args.freeze_adapters = getattr(args, 'freeze_adapters', False)
    args.freeze_layers = getattr(args, 'freeze_layers', False)
    args.freeze_embeddings = getattr(args, 'freeze_embeddings', False)
