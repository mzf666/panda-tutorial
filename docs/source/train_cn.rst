训练您的大模型（LLM)
==============

.. autosummary::
   :toctree: generated


`PandaLLM <https://github.com/dandelionsllm/pandallm>`_ 通过利用 `DeepSpeed <https://github.com/microsoft/DeepSpeed>`_ 加速框架和 `FairScale <https://github.com/facebookresearch/fairscale>`_ 并行框架，实现了各种LLM的高效训练。您可以使用以下命令使用自定义配置训练您的LLM：

.. code-block:: console

    (pandallm) $ python train.py --model llama-7b

当您执行 ``train.py`` 脚本时，它会根据位于 ``./conf/template.yaml`` 的配置模板文件自动生成位于 ``./conf/tmp.yaml`` 的训练配置文件。随后，脚本通过执行 ``./trainer_torch_fsdp_wandb.py`` 启动训练过程。如果您喜欢使用个人化配置训练您的模型，可以执行以下命令：

.. code-block:: console

    (pandallm) $ python train.py --conf_path ${PATH_TO_YOUR_CONF_FILE}

在接下来的部分中，我们将提供使用 ``train.py`` 脚本训练LLM所涉及的工作流程的全面概述。

关于 Hydra 的初步介绍
-----------------------

在此项目中，我们使用 Hydra 配合 yaml 文件来配置所有实验，包括训练和推理。虽然我们提供了一些脚本来自动生成配置文件，但您可能需要基本了解我们如何使用 Hydra 管理实验。

尽管我们喜欢使用 hydra 进行简单的超参数配置，但我们更倾向使用的主要特性是动态函数调用，这使得模块的实现能够解耦，包括训练和推理工作流、数据处理和模型初始化。
实现这一点的另一种方法是通过模块注册，就像在 ``Fairseq`` 或 ``OpenMMLab`` 中一样。然而，注册需要在一开始就加载所有已注册的模块，这将导致项目变得庞大时的高延迟，并且难以快速迭代管理。

现在，让我们看一个关于数据加载的示例。在 ``general_util.training_utils`` 中，我们使用 ``load_and_cache_examples`` 来加载数据集。然后你可以找到以下代码片段来初始化数据集：

.. code-block:: python

    dataset = hydra.utils.call(cfg, file_path=file_path, tokenizer=tokenizer)

其中 ``cfg.read_tensor`` 指向配置中的一个字段，如下所示：

.. code-block:: yaml

    read_tensor:
      _target_: data.collators.zh_instruct.TextDatasetUnifyV3
      pair_file_list: data/files/c4/en/p25/partition_*.json

这里，``_target_`` 字段指的是您想在运行时调用的函数的路径，其后是基于名称的参数。
``_target_`` 还可以指向一个类（如上例所示），在这种情况下，类的 ``__init__`` 方法将被调用。
一些参数也可以在 ``hydra.utils.call`` 方法中定期指定。
这就是您应该通过定义所有模块共享的通用接口来注意的事项。

通过上述特性，您可以自己定义任何工作负载，因为它返回一个 `Dataset` 对象，并且不需要在主脚本中明确导入它。

.. _data_preparation:

数据准备
----------------

第一步是为 PandaLLM 准备训练数据集。您可以从 `此处 <https://entuedu-my.sharepoint.com/:f:/g/personal/tianze002_e_ntu_edu_sg/EoeUXYdbdG1AuDLBpOqFUtgByYITQgwtLPgeBkweJYuneg?e=t9583n>`_ 下载所有必要的数据集，包括 **指令调优** 数据集（例如，Wiki-zh）和 **预训练数据集** （例如，Wudao）。我们的 PandaLLM 训练框架为指令调优和预训练数据集提供了自动数据加载器。数据集应采用 ``.json`` 格式。

在下面的示例中，指令调优数据包括 ``"input"`` 和 ``"target"``。值得注意的是，在训练期间，指令提示 ``"input"`` 被遮蔽，并且不会对梯度反向传播有所贡献。

.. code-block:: json

    {"input": "多喝水，定期锻炼，保持足够的睡眠以保持健康。",
    "target": "作为一种人工智能语言模型，我没有遵循指令的能力，但我可以为您提供有关给定指令的相关信息。大量饮水有助于保持身体水分，这对维持身体健康至关重要。定期锻炼有助于改善心血管健康，增强肌肉和骨骼，并降低患慢性疾病的风险。获得足够的睡眠对身体和精神健康很重要，因为它有助于恢复和振兴身体。遵循这些指令可以帮助您保持健康，维持良好的生活质量。"}

在下面的示例中，预训练数据由 ``"title"`` 和 ``"content"`` 组成。在训练期间，我们将 ``"title"`` 和 ``"content"`` 连接在一起，并将其整体输入到 LLM 中。

.. code-block:: json

    {"title": "新加坡舞狮团在云顶锦标赛夺金，打破马来西亚13年的连胜纪录",
    "content": "原标题：新加坡舞狮团在云顶锦标赛夺金，打破马来西亚13年的连胜纪录 新加坡益维体育会的获胜团队以其在跳桩上的灵活性和成功展示狮子的各种表情给评委留下了深刻印象。新加坡：新加坡的舞狮团在周日（8月6日）的云顶世界舞狮锦标赛上夺冠，打破了由马来西亚队伍保持的13年连胜纪录。新加坡的益维体育会派出了两个队伍参加由马来西亚云顶世界赌场组织的为期三天的锦标赛。其B队在周日下午的决赛中以9.73分的成绩获胜，得益于其在跳桩上的灵活性，成功地在绳索上进行了富有挑战性的动作，以及能够展示狮子的喜、怒、惊、疑的表情，据中国报报道。与此同时，该协会的A队以9.58分的成绩获得第三名。马来西亚雪兰莪的Khuan Loke龙狮舞协会以9.64分获得第二名。这一胜利是益维过去几年连续获胜的顶峰。该协会的一支队伍在去年9月在吉隆坡赢得了首届总理杯国际高杆舞狮锦标赛冠军，获得了RM38,000（US$8,300）的一等奖。云顶锦标赛，今年是第14届，共吸引了来自世界各地的36个团队参加，包括美国、法国和澳大利亚。据中国报报道，马来西亚团队在过去的13次比赛中一直保持着榜首地位。柔佛的Muar Guansheng Temple龙狮舞团获得了12个冠军，而吉打洪德体育会龙狮舞团赢得了一个。中国报还表示，获胜团队将获得15,000美元现金、奖杯和奖牌。第一和第二名将分别获得8,000美元和5,000美元现金，以及奖杯和奖牌。"}


为了兼容性，请将所有指令调优数据集存储在 ``./dataset/instruction_tuning`` 目录下，并将预训练数据集存储在 ``./dataset/pretraining`` 目录下。如果您希望使用自定义数据集训练LLM，您可以使用以下命令指定其目录：

.. code-block:: console

    (pandallm) $ python train.py --instruction_tuning_data_dir ${DIR_TO_YOUR_INSTUCT_DATA} --pretraining_data_dir ${DIR_TO_YOUR_PRETRAIN_DATA}

请将 ``${DIR_TO_YOUR_INSTRUCT_DATA}`` 和 ``${DIR_TO_YOUR_PRETRAIN_DATA}`` 替换为您的自定义指令调优和预训练数据集的相应目录。

此外，您还可以通过指定以下参数来进一步自定义数据加载器。

--num_workers  此参数确定在训练期间用于数据加载的工作进程数量。增加工作人员数量可以加速数据加载。默认值设置为 :math:`2`。

--prefetch_factor  此参数确定要预取的批次数量。预取允许数据加载器提前加载和准备下一批数据，从而减少训练期间的等待时间。默认值设置为 :math:`2`。

--max_seq_length  此参数定义训练期间输入文本的最大序列长度。任何超过此长度的输入序列将被截断或分成多个部分。默认值设置为 :math:`2048`。

.. _models:

模型
------

PandaLLM框架支持多种LLM架构，您可以使用以下的 ``--model`` 参数来指定模型类型：

.. code-block:: console

    (pandallm) $ python train.py --model ${MODEL_TYPE}

以下是支持的LLM架构。

.. list-table::
    :widths: 25 25
    :header-rows: 1

    * - 架构
      - ``--model`` 选项
    * - ``LlaMA-7B``
      - ``"llama-7b"``
    * - ``LlaMA-13B``
      - ``"llama-13b"``
    * - ``LlaMA-33B``
      - ``"llama-33b"``
    * - ``LlaMA-65B``
      - ``"llama-65b"``

您可以通过指定 ``"--ckpt_path"`` 参数，根据自定义检查点对LLM进行微调。例如，要使用最新检查点微调 ``LlaMA-7B`` 模型，请执行以下命令：

.. code-block:: console

    (pandallm) $ python train.py --model llama-7b --ckpt_path pretrain/llama-7b

该命令将启动针对 ``llama-7b`` 模型的微调过程，使用指定的 ``./pretrain/llama-7b`` 检查点。除了LlaMA检查点，您还可以从 `PandaLLM官方GitHub仓库 <https://github.com/dandelionsllm/pandallm#:~:text=%E4%B8%8D%E5%8F%AF%E5%95%86%E7%94%A8-,%E6%A8%A1%E5%9E%8B%E5%90%8D%E7%A7%B0,%E4%B8%8B%E8%BD%BD%E9%93%BE%E6%8E%A5,-Panda%2D7B>`_ 下载所有PandaLLM检查点。

要微调您的自定义LLM模型，请按照以下步骤操作：

1.  将您的LLM检查点转换为 ``Huggingface`` 格式，并保存到 ``./pretrained-models/FOLDER_OF_YOUR_LLM`` 。
#.  执行以下命令

    .. code-block:: console

        (pandallm) $ python train.py --model llama-7b --ckpt_path ${FOLDER_OF_YOUR_LLM}

    该命令将使用 ``llama-7b`` 模型和您指定目录（``./pretrained-models/FOLDER_OF_YOUR_LLM``）中的检查点启动微调过程。

优化
------------

通用设置
^^^^^^^^^^^^^^^^

PandaLLM框架为训练提供了几个功能，包括自动梯度累积，`NVLAMB <https://arxiv.org/abs/1904.00962>`_ 优化器集成，以及基于 `BitsandBytes <https://github.com/facebookresearch/bitsandbytes>`_ 的量化感知训练。要自定义训练超参数，您可以指定以下参数。下面是每个参数的描述：

--per_gpu_train_batch_size  训练期间每个GPU的批量大小。默认值为 :math:`1`。

--per_gpu_eval_batch_size  评估期间每个GPU的批量大小。默认值为 :math:`2`。

--optimizer  训练优化器。默认值为 ``"AdamW"``。

--learning_rate  训练期间模型的每批学习率。默认值为 :math:`0.001`。

--lr_scheduler  学习率调度器选项，包括 ``"linear"``, ``"cosine"``, ``"constant"``, ``"poly"``, 和 ``"warmup"``。当参数未指定时，默认值为 ``"warmup"``。

--gradient_accumulation_steps  在执行向后/更新传递之前的梯度积累步骤数。默认值为 :math:`64`。

--weight_decay  应用于模型所有参数的权重衰减。默认值为 :math:`0.00`。

--adam_epsilon  Adam优化器的 :math:`\varepsilon` 值。默认值为 :math:`10^{-6}`。

--adam_betas  在Adam优化器中用于计算梯度和平方梯度移动平均值的 :math:`\beta` 系数。默认值为 :math:`(0.9, 0.99)`。

--max_grad_norm  梯度裁剪的最大范数。默认值为 :math:`0.3`。

--num_train_epochs  训练的总时期数量。默认值为 :math:`1`。

--max_steps  最大训练步骤数。默认值为 :math:`-1`，表示没有最大限制。

--warmup_proportion  执行线性学习率预热的训练步骤的比例。默认值为 :math:`0`。

--warmup_steps  学习率预热的预热步数。默认值为 :math:`50`。

--bit_training  这个 ``boolean`` 参数指定了量化感知训练的位训练模式。它决定了训练过程中权重和激活的精度。默认值为 ``False``。

要以 :math:`0.002` 的学习率对 ``Panda-7B`` 模型进行 :math:`2` 个时期的微调，请执行以下命令：

.. code-block:: console

        (pandallm) $ python train.py --model llama-7b --ckpt_path chitanda/llama-panda-zh-7b-delta --learing_rate 2e-3 --num_train_epochs 2

低秩适应 (LoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^

PandaLLM支持使用 `LoRA <https://github.com/huggingface/peft>`_ 微调LLM。例如，要使用LoRA启动 ``LlaMA-65B`` 模型的训练过程，请执行以下命令：

.. code-block:: console

        (pandallm) $ python train.py --model llama-65b --use_lora --lora_r 64 --lora_alpha 16 --lora_dropout 0.05

您可以通过指定以下参数来自定义LoRA在LLM训练过程中的行为。

--use_lora  此 ``boolean`` 参数在训练过程中启用 LoRA 。指定后，LoRA将整合到LLM的训练中。

--lora_r  此参数确定在LoRA适应期间每个令牌所考虑的本地邻居数量。默认值设置为 :math:`64`。

--lora_alpha  此参数控制LoRA的适应强度。它影响模型对局部关联的适应程度。默认值设置为 :math:`16`。

--lora_dropout  此参数指定在LoRA适应期间应用的退出率。退出有助于规范训练过程并防止过度拟合。默认值设置为 :math:`0.05`。


Quantization-aware training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

PandaLLM基于 `BitsandBytes <https://github.com/facebookresearch/bitsandbytes>`_ 框架启用Quantization-aware training。例如，要使用 `BitsandBytes` 量化方案训练具有 :math:`4` 位精度的 ``LlaMA-65B`` 模型，请执行以下命令：

.. code-block:: console

        (pandallm) $ python train.py --model llama-65b --use_quant
