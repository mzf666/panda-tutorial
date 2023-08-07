流水线并行
====================

初步
-----------

流水线并行（PP）是一种先进的技术，使得有效的模型并行训练成为可能，其中不同的层被放置在不同的GPU上，并且前向和后向传递以流水线方式执行，
这可以减少每个GPU的内存使用并与简单模型并行相比大大提高GPU利用率。

与使用 ZeRO-2/3 与卸载训练模型的极端情况相比，流水线并行可以大大提高训练效率并减少内存使用。
在我们的 Panda 项目中，我们为流行的开源模型 LLaMA 和 MPT 提供了两个示例实现。由于PP需要对原始模型实现进行特定修改，我们无法覆盖所有模型。因此，本教程的一个重要目标是提供一个模板以及一些说明，以便您可以快速将其适应自己的用途并专注于模型的实现。

通过前面的部分，我们相信您已经对我们的训练流程以及基于 hydra 的动态配置如何工作有了一般的了解。
为了将PP适应到您自己的情况，您通常可以按照以下步骤操作：
1. 创建自己的数据处理管道。
2. 为任何未包括的模型创建特定的流水线并行包装。
3. 运行它。

在以下部分，我们将介绍对主要训练流程的修改，并解决您在实现自己的模型过程中可能遇到的问题。

核心代码片段
------------------

模型实现
^^^^^^^^^^^^^^^^^^^^

首先，目前所有的流水线并行实现都要求您使用 `nn.Sequential` 来重新组织我们的模型，输入/输出应该是元组。
这用于异步前向和后向传递。实现这一点的最简单方法是添加一个简单的包装器来继承Transformer层，并重写 ``forward`` 函数
用于输入解包和输出打包。例如，LLaMA层的代码片段如下：


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

同样，您也可以为 ``nn.Embedding`` 和 ``LayerNorm`` 实现 Wrapper ，以便最后一层看起来像下面这样：

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


其中 ``LayerSpec`` 是DeepSpeed为后初始化提供的特殊类，我们将在下一节中介绍它。

对于损失函数，您可以定义一个继承 ``nn.Module`` 的类并将其添加到 ``nn.Sequential`` 或 ``List`` 中，
或者定义一个可调用函数。
这两种方法的区别在于，前者的输入仍然是模型最后一层的元组输出。
在这种情况下，您应该从第一层传递到最后一层的 ``labels`` 。
对于第二个，损失函数的输入是一个 ``(outputs, labels)`` 的元组，其中 ``outputs`` 来自模型的最后一层，
``labels`` 直接来自数据加载器。我们为这两种方法提供了两个示例情况：

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


无论您使用哪种方法，collator的返回值应该如下所示：
              
.. code-block:: python

    return (
            (input_ids, attention_mask, other_inputs),  # The inputs to the first layer
            labels,  # The labels, and will be passed to the loss function.
        )

它确实是一个元组包含元组。对于第二种情况，您应该在 DeepSpeed ``PipelineModule`` 中指定损失函数，如下所示：
              
.. code-block:: python

    model_pipe = PipelineModule(layers=layers,
                                num_stages=cfg.num_stages,
                                loss_fn=pp_loss_fn,  # Specify the callable loss function here.
                                partition_method=getattr(cfg, "partition_method", "parameters"),
                                activation_checkpoint_interval=getattr(cfg, "activation_checkpoint_interval", 0)
                                )




模型参数初始化
^^^^^^^^^^^^^^^^^^^^

有两种主要方法来启用模型初始化并加载预训练权重。一种是首先使用 ``from_pretrained`` 函数初始化模型。
在这种情况下，您可以参考 ``models.llama_ds_mp_wrap.get_model`` 了解详情。
这种方法的缺点是它会为每个工作器加载整个模型。当模型很大时，这将导致CPU内存耗尽。
另一种方法是首先使用DeepSpeed的 ``LayerSpec`` 类初始化分片模型，以在管道并行分区后实施初始化。
然后，每个等级只需要为每个自己的分区加载预训练权重：


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


注意，预训练的权重应该通过使用 ``convert2ckpt.py`` 从HF格式转换。


### 管道并行（PP）和分布式数据并行（DP）的混合训练

当 ``dist.world_size > num_stages`` 时，将自动启用混合训练。管道并行（PP）的阶段数是 ``num_stages``
而数据并行（DP）的程度是 ``dist.world_size // num_stages``。

### 无权重类型的词嵌入

与传统的预训练语言模型不同，LLaMA 不需要权重类型化。因此，不要使用 ``TiedLayerSpec`` 来包装 ``embed_tokens`` 和 ``lm_head`` 模块。

``MPT`` 的实现已经包括了权重类型化，您可以参考以了解详情。

### 分布式采样器设置

当启用PP和DP的混合训练时，应该小心为每个等级设置其状态（PP阶段和DP组）的 ``DistributedSampler``。

核心代码片段如下：

.. code-block:: python

    dp_degree = dist.get_world_size() // cfg.num_stages

    if dp_degree > 1:
        dp_id = model.grid.get_data_parallel_id()
        sub_train_sampler = DistributedSampler(sub_train_dataset, num_replicas=dp_degree, rank=dp_id)
    else:
        sub_train_sampler = RandomSampler(sub_train_dataset)


DeepSpeed和CPU内存减少的数据获取设计
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在DeepSpeed设计中，特定PP组中，只有第一个和最后一个等级，即 ``stage=0 or stage=num_stages - 1``，
会从dataloader获取小批量数据，而其他等级永远不会获取数据。

基于此，对于dataloader永远不会使用的等级，我们可以使用占位符来分配内存使用。当训练大型模型时，这可能特别有用。
例如，当使用 ``offload_optimizer=True`` 和 ``num_stages=8`` 训练LLaMA-65B时，CPU内存使用量已经接近800GB，
当您使用大型数据集时，这将导致CPU内存溢出。

数据集占位符的代码如下：

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

其中 ``TestDataset`` 是一个空数据集，collator是符合输入格式的任意一个。

已知问题和可能的解决方案
------------------------------------

BF16支持
^^^^^^^^^^^^

通过在deepspeed配置中设置以下内容，可以使用Bfloat16：

.. code-block:: yaml

    data_types:
      grad_accum_dtype: "fp32"


然而，bfloat16不能与优化器offload一起使用。请注意，管道并行被设计为不支持优化器offload（请参阅问题[\#3866](https://github.com/microsoft/DeepSpeed/issues/3866)）。尽管如此，在fp16训练下仍可以启用。

..
 ### Flash Attention

我无法使用原始实现或pytorch 2.0中的 `torch.nn.functional.scaled_dot_product_attention` 启用 Flash Attention。参见问题[此处](https://github.com/HuangLK/llama-deepspeed/issues/36)和[此处](https://github.com/microsoft/DeepSpeed/issues/3868)。

Torch Compile
^^^^^^^^^^^^^

模板中不支持Torch Compile，这可能是因为我写得不正确。请多指正。

参考和致谢
---------------------------

1. [llama-deepspeed](https://github.com/HuangLK/llama-deepspeed/tree/main)
2. [ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning)
3. [DeepSpeed Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/)

          
