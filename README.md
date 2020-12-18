# Multilingual-Adapters

## Introduction

We provide an implementation of 

This repo contains the code to replicate all experiments from


## Training a multilingual model with adapters

Our methods introduce new hyper-parameters

+ --freeze-layers which sets 
+ --freeze-embeddings
+ --languages-adapters
+ --lang-pairs-adapters
+ --adapters-type
+ --adapter-projection-dim

Below is an example of training with adapters


```bash
DATA_BIN=<path to binarized data>
CHECKPOINTS_SHARED_MODEL=<path to the checkpoint of the fully shared model>
SAVE_DIR=< >

LANG_PAIRS ="en-bg,en-cs,en-da,en-de,en-el,en-es,en-et,en-fi,en-fr,en-hu,en-it,en-lt,en-lv,en-nl,en-pl,en-pt,en-ro,en-sk,en-sl,en-sv,en-mt,en-hr,en-ga,bg-en,cs-en,da-en,de-en,el-en,es-en,et-en,fi-en,fr-en,hu-en,it-en,lt-en,lv-en,nl-en,pl-en,pt-en,ro-en,sk-en,sl-en,sv-en,mt-en,hr-en,ga-en"

ADAPTERS_DIM=<hidden dimension of the adapters>
ADAPTERS_CONDITION=<language-pairs, source, target, source+target>
LANGUAGES_ADAPTERS=< >

fairseq-train ${DATA_BIN} \
    --fp16 \
    --ddp-backend=no_c10d \
    --task multilingual_translation \
    --lang-pairs  \
    --arch multilingual_adapters \
    --share-encoders \
    --share-decoders \
    --share-decoder-input-output-embed \
    --adapter-projection-dim ${ADAPTERS_DIM} \
    --freeze-embeddings \
    --freeze-layers \
    --adapters-type ${ADAPTERS_CONDITION} \
    --languages-adapters ${LANGUAGES_ADAPTERS} \
    --reset-optimizer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' \
    --warmup-updates 16000 \
    --warmup-init-lr '1e-07' \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --max-tokens 4096 \
    --encoder-langtok tgt \
    --restore-file ${CHECKPOINTS_SHARED_MODEL} \
    --save-dir ${SAVE_DIR} \
```


## Inference command



```bash
DATA_BIN=<path to binarized data>
CHECKPOINT=<path to the checkpoint of the fully shared model>
RESULTS=<path to binarized data>

LANG_PAIRS ="en-bg,en-cs,en-da,en-de,en-el,en-es,en-et,en-fi,en-fr,en-hu,en-it,en-lt,en-lv,en-nl,en-pl,en-pt,en-ro,en-sk,en-sl,en-sv,en-mt,en-hr,en-ga,bg-en,cs-en,da-en,de-en,el-en,es-en,et-en,fi-en,fr-en,hu-en,it-en,lt-en,lv-en,nl-en,pl-en,pt-en,ro-en,sk-en,sl-en,sv-en,mt-en,hr-en,ga-en"
SOURCE_LANG=
TARGET_LANG


        fairseq-generate $DATA_BIN --task multilingual_translation \
        --lang-pairs ${LANG_PAIRS} \
        --path ${CHECKPOINT} \
        --source-lang ${SOURCE_LANG} \
        --target-lang ${TARGET_LANG} \
        --model-overrides "{'source_lang':'${SOURCE_LANG}','target_lang':'${TARGET_LANG}'}" \
        --beam 5 \
        --encoder-langtok tgt \
        --results-path ${RESULTS} \
        --remove-bpe 
```


## Citation
```bibtex
@article
}
```
