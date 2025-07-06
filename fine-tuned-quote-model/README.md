---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:2508
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: “the only truth is music.” - jack kerouac [music, truth]
  sentences:
  - “reader's bill of rights1. the right to not read 2. the right to skip pages 3.
    the right to not finish 4. the right to reread 5. the right to read anything 6.
    the right to escapism 7. the right to read anywhere 8. the right to browse 9.
    the right to read out loud 10. the right to not defend your tastes”
  - “i would rather walk with a friend in the dark, than alone in the light.”
  - “the only truth is music.”
- source_sentence: “maybe...you'll fall in love with me all over again.""hell," i
    said, "i love you enough now. what do you want to do? ruin me?""yes. i want to
    ruin you.""good," i said. "that's what i want too.” - ernest hemingway, [infatuation,
    love]
  sentences:
  - “maybe...you'll fall in love with me all over again.""hell," i said, "i love you
    enough now. what do you want to do? ruin me?""yes. i want to ruin you.""good,"
    i said. "that's what i want too.”
  - “ah, music," he said, wiping his eyes. "a magic beyond all we do here!”
  - “maybe this world is another planetâ€™s hell.”
- source_sentence: “i must learn to be content with being happier than i deserve.”
    - jane austen, [happiness]
  sentences:
  - “a paranoid is someone who knows a little of what's going on. ”
  - “i must learn to be content with being happier than i deserve.”
  - “i hope you will have a wonderful year, that you'll dream dangerously and outrageously,
    that you'll make something that didn't exist before you made it, that you will
    be loved and that you will be liked, and that you will have people to love and
    to like in return. and, most importantly (because i think there should be more
    kindness and more wisdom in the world right now), that you will, when you need
    to be, be wise, and that you will always be kind.”
- source_sentence: “not all of us can do great things. but we can do small things
    with great love.” - mother teresa [misattributed-to-mother-teresa, paraphrased]
  sentences:
  - “i read once that the ancient egyptians had fifty words for sand & the eskimos
    had a hundred words for snow. i wish i had a thousand words for love, but all
    that comes to mind is the way you move against me while you sleep & there are
    no words for that.”
  - “i shut my eyes and all the world drops dead; i lift my eyes and all is born again.”
  - “not all of us can do great things. but we can do small things with great love.”
- source_sentence: “i believe the universe wants to be noticed. i think the universe
    is inprobably biased toward the consciousness, that it rewards intelligence in
    part because the universe enjoys its elegance being observed. and who am i, living
    in the middle of history, to tell the universe that it-or my observation of it-is
    temporary?” - john green, [john-green, tfios, the-fault-in-our-stars]
  sentences:
  - “don't you think it's better to be extremely happy for a short while, even if
    you lose it, than to be just okay for your whole life?”
  - '“here''s all you have to know about men and women: women are crazy, men are stupid.
    and the main reason women are crazy is that men are stupid.”'
  - “i believe the universe wants to be noticed. i think the universe is inprobably
    biased toward the consciousness, that it rewards intelligence in part because
    the universe enjoys its elegance being observed. and who am i, living in the middle
    of history, to tell the universe that it-or my observation of it-is temporary?”
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '“i believe the universe wants to be noticed. i think the universe is inprobably biased toward the consciousness, that it rewards intelligence in part because the universe enjoys its elegance being observed. and who am i, living in the middle of history, to tell the universe that it-or my observation of it-is temporary?” - john green, [john-green, tfios, the-fault-in-our-stars]',
    '“i believe the universe wants to be noticed. i think the universe is inprobably biased toward the consciousness, that it rewards intelligence in part because the universe enjoys its elegance being observed. and who am i, living in the middle of history, to tell the universe that it-or my observation of it-is temporary?”',
    "“don't you think it's better to be extremely happy for a short while, even if you lose it, than to be just okay for your whole life?”",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9852, 0.0722],
#         [0.9852, 1.0000, 0.0934],
#         [0.0722, 0.0934, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2,508 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                         |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             |
  | details | <ul><li>min: 18 tokens</li><li>mean: 56.14 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 41.81 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                   |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>“two people in love, alone, isolated from the world, that's beautiful.” - milan kundera [love]</code>                                                                                                                                                                                                                                                     | <code>“two people in love, alone, isolated from the world, that's beautiful.”</code>                                                                                                                                                                                                                         |
  | <code>“fantasy is escapist, and that is its glory. if a soldier is imprisioned by the enemy, don't we consider it his duty to escape?. . .if we value the freedom of mind and soul, if we're partisans of liberty, then it's our plain duty to escape, and to take as many people with us as we can!” - j.r.r. tolkien [fantasy, literature, philosophy]</code> | <code>“fantasy is escapist, and that is its glory. if a soldier is imprisioned by the enemy, don't we consider it his duty to escape?. . .if we value the freedom of mind and soul, if we're partisans of liberty, then it's our plain duty to escape, and to take as many people with us as we can!”</code> |
  | <code>“maybe our girlfriends are our soulmates and guys are just people to have fun with.” - candace bushnell, [dating, humor, relationships, soulmates]</code>                                                                                                                                                                                                 | <code>“maybe our girlfriends are our soulmates and guys are just people to have fun with.”</code>                                                                                                                                                                                                            |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.13.1
- Sentence Transformers: 5.0.0
- Transformers: 4.53.1
- PyTorch: 2.7.1+cpu
- Accelerate: 1.8.1
- Datasets: 3.6.0
- Tokenizers: 0.21.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->