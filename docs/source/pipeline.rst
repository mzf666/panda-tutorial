Pipeline Parallelism
=============================================================

Preliminary
-----------

Pipeline parallelism (PP) is an advanced technique enabling efficient model parallel training, where different layers are put on different GPUs and the forward and backward passes are executed in a pipelined fashion,
which can reduce the memory usage of each GPU and greatly improve the GPU utilization rate compared with naive model parallelism.

Compared with the extreme case that using ZeRO-2/3 with offload to train models, PP can greatly improve the training efficiency and reduce the memory usage.
In our Panda Project, we provide two example implementations for popular open-sourced models, LLaMA and MPT. Since PP requires specific modification to original model implementation,
we cannot cover models. So one important goal of this tutorial is to provide a template as well as some instructions so that you can quickly adapt it to your own usage and focus on the implementation of model.

Through the former sections, we believe you have already a general understanding of our training pipeline and how hydra-based dynamic configuration works.
In order to adapt PP into your own case, you can generally follow the following step:
1. Create your own data processing pipeline.
2. Create the specific pipeline parallel wrapping for any not included model.
3. Run it.

And in the following sections, we will introduce the modification of the main training loop compared with the former one,
and tackle the possible problems you may encounter during implementing your onw model.

Core Code Snippets
----------------------

### Model Implementation

First of all, currently all pipeline parallelism implementation requires you to use `nn.Sequential` to re-organize our model, and the inputs/outputs should be tuple.
This is used for asynchronous forward and backward passes. The easiest way to do this is adding a simple wrapper to inherit Transformer layer and override the ``forward`` function
for inputs unpacking and outputs packing. For example, the code snippet of LLaMA layer is as follows:

.. code-block:: python

    class ParallelTransformerLayerPipe(LlamaDecoderLayer):
        def __init__(self, config: LlamaConfig, activation_checkpointing: bool = False):
            super().__init__(config)
            self.activation_checkpointing = activation_checkpointing

        def forward(self, args):
            if self.activation_checkpointing:
                return self._ckpt_forward(args)

            hidden_states, attention_mask, position_ids = args
            outputs = LlamaDecoderLayer.forward(self,
                                                hidden_states,
                                                attention_mask,
                                                position_ids,
                                                )
            return outputs[0], attention_mask, position_ids

        def _ckpt_forward(self, args):
            hidden_states, attention_mask, position_ids = args

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return LlamaDecoderLayer.forward(module, *inputs)

                return custom_forward

            # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
            outputs = deepspeed.checkpointing.checkpoint(
                create_custom_forward(self),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )

            return outputs, attention_mask, position_ids

Similarly, you can also implement the wrapper for ``nn.Embedding`` and ``LayerNorm``, so that the final layers are like the following:

.. code-block:: python

    def get_layers_from_config(model_config, activation_checkpointing: bool = False):
        """
        `tie_word_embeddings` in LLaMA is set to `false`.
        """
        layers = [
            LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
            *[LayerSpec(ParallelTransformerLayerPipe, model_config, activation_checkpointing)
              for _ in range(model_config.num_hidden_layers)],
            LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
            LayerSpec(LMLayerPipe, model_config.hidden_size, model_config.vocab_size, bias=False),
        ]
        return layers

where ``LayerSpec`` is a special class provided by DeepSpeed for post-initialization and we will introduce it in the next section.

For loss function, you can either define a class inheriting ``nn.Module`` and add it to ``nn.Sequential`` or ``List`` directly,
or defining a callable function.
The difference between these two approaches is that the input to the former one is still the tuple output of the last layer of model.
In this case, you should pass ``labels`` from the first layer to the last layer.
For the second one, the inputs to the loss function is a tuple of ``(outputs, labels)``, where ``outputs`` is the from the last layer of the model,
and ``labels`` directly come from the data loader. We provided two examples cases for the two approaches:

.. code-block:: python

    # nn.Module based approach
    class LossLayer(torch.nn.Module):
        def forward(self, args):
            logits, labels = args
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            return loss

    # Function based approach
    def loss_fn(outputs, labels):
        logits = outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        return loss


And no matter which method you use, the return value of the collator should be like the following:

.. code-block:: python

    return (
            (input_ids, attention_mask, other_inputs),  # The inputs to the first layer
            labels,  # The labels, and will be passed to the loss function.
        )

It's indeed a tuple over tuple. And for the second case, you should specify the loss function at the DeepSpeed ``PipelineModule`` like:

.. code-block:: python

    model_pipe = PipelineModule(layers=layers,
                                num_stages=cfg.num_stages,
                                loss_fn=pp_loss_fn,  # Specify the callable loss function here.
                                partition_method=getattr(cfg, "partition_method", "parameters"),
                                activation_checkpoint_interval=getattr(cfg, "activation_checkpoint_interval", 0)
                                )




### Model initialization

There are two main approaches to enable model initialization and loading pre-trained weights. One is first initializing the model using the ``from_pretrained`` function.
In this case, you may refer to ``models.llama_ds_mp_wrap.get_model`` for details.
The drawback of this method is that it will load the whole model for each worker. This will cause out-of-CPU-memory-usage when the model is large.
Another method is first initializing the sharded models with DeepSpeed's ``LayerSpec`` class to implement post-initialization after pipeline parallelism partition.
Then each rank only need to load the pre-trained weights for each own partition:

.. code-block:: python

    model_or_config = transformers.AutoConfig.from_pretrained(cfg.model_name_or_path)
    layers = models.llama_ds_mp_wrap.get_layers_from_config(model_or_config)
    model_pipe = PipelineModule(layers=layers,
                                num_stages=cfg.num_stages,
                                loss_fn=models.llama_ds_mp_wrap.loss_fn,
                                activation_checkpoint_interval=getattr(cfg, "activation_checkpoint_interval", 0)
                                )


    ...
    model.load_checkpoint(cfg.model_name_or_path, load_module_only=True, load_optimizer_states=False, load_lr_scheduler_states=False)


Note that the pre-trained weights should be converted from HF format by using ``convert2ckpt.py``.


### Hybrid Training of Pipeline Parallelism (PP) and Distributed Data Parallel (DP)

When ``dist.world_size > num_stages``, hybrid training is automatically enabled. The number of stages of pipeline parallel (PP) is ``num_stages``
while the degree of data-parallel (DP) is ``dist.world_size // num_stages``.

### No Weight Typing of Word Embedding

Different from traditional pre-trained language models, LLaMA do not need weight typing. So do not use ``TiedLayerSpec`` to wrap ``embed_tokens`` and ``lm_head`` modules.

The implementation of ``MPT`` has included weight typing and you can refer to it for details.

### Distributed Sampler Setting

When hybrid training of PP and DP is enabled, ``DistributedSampler`` should be carefully set for each rank w.r.t. its state (PP stage and DP group).

The core code snippet is as follows:

.. code-block:: python

    dp_degree = dist.get_world_size() // cfg.num_stages

    if dp_degree > 1:
        dp_id = model.grid.get_data_parallel_id()
        sub_train_sampler = DistributedSampler(sub_train_dataset, num_replicas=dp_degree, rank=dp_id)
    else:
        sub_train_sampler = RandomSampler(sub_train_dataset)


### Data Fetch Design of DeepSpeed and CPU Memory Reduction

In DeepSpeed design, among specific PP group, only the first and the last rank, i.e., ``stage=0 or stage=num_stages - 1``,
will fetch minibatch from dataloader, and the other ranks never fetch data.

Based on this, for the ranks where the dataloader will never be used, we can use placeholders to allocate the memory usage. This could be especially useful when training large models.
For example, when training LLaMA-65B with ``offload_optimizer=True`` and ``num_stages=8``, the CPU memory usage is already nearly 800GB,
which will cause CPU memory OOM when you are using large dataset.

The code of dataset placeholder is as follows:

.. code-block:: python

    def load_empty_dataset_and_collator(cfg: DictConfig):
        from data.test import TestDataset
        from data.flan import FlanCollatorOverCollator

        dataset = TestDataset(None, None, getattr(cfg, "total_dataset_len", -1))
        collator = FlanCollatorOverCollator(collator=None,
                                            tokenizer=cfg.model_name_or_path,
                                            max_seq_length=128,
                                            decoder_only=True,
                                            return_standard_inputs=True,
                                            )

        # Keep consistent with `load_and_cache_examples`.
        if getattr(cfg, "dist_load_data_barrier", True):
            dist.barrier()

        if dist.is_initialized():
            dist.barrier()

        return dataset, collator

    if model.is_first_stage() or model.is_last_stage():
        sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)

        if dp_degree > 1:
            dp_id = model.grid.get_data_parallel_id()
            sub_train_sampler = DistributedSampler(sub_train_dataset, num_replicas=dp_degree, rank=dp_id)
        else:
            sub_train_sampler = RandomSampler(sub_train_dataset)
        sub_train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None

        sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                          sampler=sub_train_sampler,
                                          batch_size=cfg.train_batch_size,
                                          collate_fn=sub_train_collator,
                                          num_workers=cfg.num_workers,
                                          pin_memory=True,
                                          prefetch_factor=cfg.prefetch_factor,
                                          drop_last=True,
                                          )
    else:
        sub_train_dataset, sub_train_collator = load_empty_dataset_and_collator(cfg)
        sub_train_sampler = None

        sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                          batch_size=cfg.train_batch_size,
                                          collate_fn=sub_train_collator,
                                          drop_last=True,
                                          shuffle=False)



where ``TestDataset`` is an empty dataset and the collator is arbitrary one meeting the input format.

## Know Problems and Possible Solutions

### BF16 Support

Bfloat16 can be used by setting the following in deepspeed config:

.. code-block:: yaml

    data_types:
      grad_accum_dtype: "fp32"


However, bfloat16 cannot be used with optimizer offload. Note that pipeline parallelism is designed not to support optimizer offload (see issue [\#3866](https://github.com/microsoft/DeepSpeed/issues/3866)). Nevertheless, it can still be enabled under fp16 training.

### Flash Attention

I cannot enable flash attention using both the original implementation or `torch.nn.functional.scaled_dot_product_attention` from pytorch 2.0. See issue [here](https://github.com/HuangLK/llama-deepspeed/issues/36) and [here](https://github.com/microsoft/DeepSpeed/issues/3868).

### Torch Compile

Torch compilation is not supported in the template, which perhaps becuase my writing is incorrect.

## Reference & Acknowledgement

1. [llama-deepspeed](https://github.com/HuangLK/llama-deepspeed/tree/main)
2. [ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning)
3. [DeepSpeed Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/)

[//]: # (### Quick Notes)

[//]: # ()
[//]: # (#### Data fetech)

[//]: # ()
[//]: # (1. Currently most implementations uses `shuffle=True` instead of `DistributedSampler` or `RandomSampler` of pytorch in data loader. I find that for `wordld_size=4` scenario, only the first rank and the last one fetech data from data loader. This can be verified by adding print information in `__getitem__` method of specific dataset. However, when really training, I find that only the batch feteched from the first rank will be really send to model. This is consistent with what I thought about pipeline parallelism that only one rank feteches data and the other ranks only take the outputs from the previous rank as iputs.)

[//]: # (2. There is a bug in Deepspeed hybrid engine loading model checkpoint that there mush be optimizer states in the specific dir, check it [here]&#40;https://github.com/HuangLK/llama-deepspeed/issues/28&#41;.)
