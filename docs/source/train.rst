Train your LLM
==============

.. autosummary::
   :toctree: generated


`PandaLLM <https://github.com/dandelionsllm/pandallm>`_ enables efficient training of various LLMs by leveraging the `DeepSpeed <https://github.com/microsoft/DeepSpeed>`_ acceleration framework and the `FairScale <https://github.com/facebookresearch/fairscale>`_ parallelization framework. You can train your LLM with a customized configuration using the following command:

.. code-block:: console

    (pandallm) $ python train.py --model llama-7b

When you execute the ``train.py`` script, it automatically generates a training configuration file at ``./conf/tmp.yaml`` based on the configuration template file located at ``./conf/template.yaml``. Subsequently, the script initiates the training process by executing ``./trainer_torch_fsdp_wandb.py``. If you prefer to train your model with a personalized configuration, you can execute the following command:

.. code-block:: console

    (pandallm) $ python train.py --conf_path ${PATH_TO_YOUR_CONF_FILE}

In the forthcoming sections, we provide a comprehensive overview of the workflow involved in training an LLM using the ``train.py`` script.

Preliminary about Hydra
-----------------------

In this project, we use Hydra with yaml file to configure all experiments, including both training and inference. Although we have provided
some scripts to automatically generate config file, you may need to have a basic understanding how we use Hydra to manage our experiments.

Despite simple hyper-parameter configuration, the main feature we prefer to use hydra is dynamically function calling, which enables decoupled
module implements, including training & inference workflow, data processing, and model initialization.
Another approach to implement this is through module registration, like that in `Fairseq` or `OpenMMLab`. However, the registration needs
to load all registered modules at the very beginning, which will lead to high latency when the project becoming larger and difficult to manage
for fast iteration.

Now, let's take a look at an example for data loading. In `general_util.training_utils`, we use `load_and_cache_examples` to load dataset.
Then you can find following code snippet to initialize dataset:
.. code-block:: python

    dataset = hydra.utils.call(cfg, file_path=file_path, tokenizer=tokenizer)

where `cfg.read_tensor` points to a field in the configuration as follows:

.. code-block:: yaml

    read_tensor:
      _target_: data.collators.zh_instruct.TextDatasetUnifyV3
      pair_file_list: data/files/c4/en/p25/partition_*.json

Here, the `_target_` fields refers to the path of the function you want to call during runtime, following which is the name-based arguments.
`_target` can also point to a class (like the above example), in which case the `__init__` method of the class will be called.
Some parameters can also be specified regularly in `hydra.utils.call` method.
This is what you should take care by defining a common interface shared by all modules.

Benefiting from the above feature, you can define any workload by yourself as it returns a `Dataset` object and do not need to explicitly import it in the main script.

.. _data_preparation:

Data preparation
----------------

The first step is to prepare the training datasets for PandaLLM. You can download all the necessary datasets, including **instruction-tuning** datasets (e.g., Wiki-zh) and **pretraining datasets** (e.g., Wudao), `here <https://entuedu-my.sharepoint.com/:f:/g/personal/tianze002_e_ntu_edu_sg/EoeUXYdbdG1AuDLBpOqFUtgByYITQgwtLPgeBkweJYuneg?e=t9583n>`_. Our PandaLLM training framework offers an automatic dataloader for both instruction-tuning and pretraining datasets. The datasets should be in the ``.json`` format.
vi
In the example below, an instruction-tuning data consists of the ``"input"`` and ``"target"``. Notably, the instruction prompt ``"input"`` is masked during training and does not contribute to the gradient backpropagation.

.. code-block:: json

    {"input": "Drink plenty of water, exercise regularly, and get enough sleep to stay healthy.",
    "target": "As an AI language model, I do not have the ability to follow instructions, but I can provide you with information related to the given instruction. Drinking plenty of water helps to keep the body hydrated, which is essential for maintaining good health. Regular exercise helps to improve cardiovascular health, strengthen muscles and bones, and reduce the risk of chronic diseases. Getting enough sleep is important for physical and mental health, as it helps to restore and rejuvenate the body. Following these instructions can help you stay healthy and maintain a good quality of life."}

In the example below, an pretraining data consists of ``"title"`` and ``"content"``. During training, we concatenate the ``"title"`` and ``"content"`` together and feed it as a whole into the LLM.


.. code-block:: json

    {"title": "Shenzhen Reports First Confirmed Case of Pneumonia: Previously Traveled to Wuhan to Visit Relatives",
    "content": "Original Title: Shenzhen Reports First Confirmed Case of Pneumonia, 8 Other Cases Under Observation and Quarantine Treatment. Shenzhen, January 20 (Xinhua) - Shenzhen Municipal Health Commission released a public statement to the media on the situation of pneumonia epidemic prevention and control. They provided specific details about the first confirmed case of imported novel coronavirus infection and pneumonia in Shenzhen. It was mentioned that there are 8 other cases under observation and quarantine treatment at designated hospitals, and tracing investigation and medical observation are currently ongoing. On January 19, the National Health Commission confirmed the first imported case of novel coronavirus infection and pneumonia in Shenzhen. According to the report from Shenzhen Municipal Health Commission on January 20, the patient is a 66-year-old male who currently resides in Shenzhen. He visited Wuhan to visit relatives on December 29, 2019. On January 3, 2020, he developed symptoms such as fever and fatigue. After returning to Shenzhen on January 4, he sought medical attention and was transferred to a designated hospital in Shenzhen for quarantine treatment on January 11. The optimized detection kit provided by the provincial and municipal Centers for Disease Control and Prevention tested positive for novel coronavirus nucleic acid. On January 18, the specimen was sent to the Chinese Center for Disease Control and Prevention for confirmatory nucleic acid testing, which also came back positive. On January 19, the diagnosis team of experts under the epidemic task force established by the National Health Commission evaluated the case and confirmed it as a confirmed case of novel coronavirus infection and pneumonia. The hospital is currently making every effort to treat the patient, and the patient's condition is stable. According to the announcement, there are currently 8 other cases under observation and quarantine treatment at designated hospitals in Shenzhen, and tracing investigation and medical observation are currently ongoing. Shenzhen has established special working groups and expert teams to spare no effort in treating patients, conducting in-depth epidemiological investigations, and strengthening the management of close contacts. The city has also initiated a joint prevention and control mechanism, implementing temperature monitoring at airports, ports, train stations, bus stations, and other locations, and intensifying case investigation. Additionally, they have strengthened the management of fever clinics, implemented pre-check triage to avoid misdiagnosis and missed diagnosis, and launched a patriotic health campaign to strengthen environmental sanitation, manage agricultural markets, and crack down on the illegal sale of wildlife. Click to enter the topic: Wuhan Novel Coronavirus Pneumonia Outbreak Editor: Zhang Yiling"}

For compatibility purposes, please store all instruction-tuning datasets under the ``./dataset/instruction_tuning`` directory, and pretraining datasets under the ``./dataset/pretraining`` directory. If you wish to train LLMs with a custom dataset, you can specify its directory using the following command:

.. code-block:: console

    (pandallm) $ python train.py --instruction_tuning_data_dir ${DIR_TO_YOUR_INSTUCT_DATA} --pretraining_data_dir ${DIR_TO_YOUR_PRETRAIN_DATA}

Please replace ``${DIR_TO_YOUR_INSTRUCT_DATA}`` and ``${DIR_TO_YOUR_PRETRAIN_DATA}`` with the respective directories for your custom instruction-tuning and pretraining datasets.

Additionally, you can further customize the dataloader by specifying the following arguments.

--num_workers  This argument determines the number of worker processes to use for data loading during training. Increasing the number of workers can accelerate data loading. The default value is set to :math:`2`.

--prefetch_factor  This argument determines the number of batches to prefetch. Prefetching allows the dataloader to load and prepare the next batches in advance, reducing the waiting time during training. The default value is set to :math:`2`.

--max_seq_length  This argument defines the maximum sequence length allowed for input texts during training. Any input sequence exceeding this length will be truncated or split into multiple parts. The default value is set to :math:`2048`.



.. _models:

Models
------

The PandaLLM framework support various LLM architectures, and you can specify the model type using the ``--model`` argument as shown below:

.. code-block:: console

    (pandallm) $ python train.py --model ${MODEL_TYPE}

Here are the supported LLM architectures.

.. list-table::
    :widths: 25 25
    :header-rows: 1

    * - Architectures
      - ``--model`` options
    * - ``LlaMA-7B``
      - ``"llama-7b"``
    * - ``LlaMA-13B``
      - ``"llama-13b"``
    * - ``LlaMA-33B``
      - ``"llama-33b"``
    * - ``LlaMA-65B``
      - ``"llama-65b"``

You can finetune a LLM based on a custom checkpoint by specifying the ``"--ckpt_path"`` argument. For example, to finetune a ``LlaMA-7B`` model using the latest checkpoint, execute the following command:

.. code-block:: console

    (pandallm) $ python train.py --model llama-7b --ckpt_path pretrain/llama-7b

This command will initiate the fine-tuning process for the ``llama-7b`` model, utilizing a specified ``./pretrain/llama-7b`` checkpoint. Beside the LlaMA checkpoints, you can also download all the PandaLLM checkpoints from the `official PandaLLM GitHub repository <https://github.com/dandelionsllm/pandallm#:~:text=%E4%B8%8D%E5%8F%AF%E5%95%86%E7%94%A8-,%E6%A8%A1%E5%9E%8B%E5%90%8D%E7%A7%B0,%E4%B8%8B%E8%BD%BD%E9%93%BE%E6%8E%A5,-Panda%2D7B>`_.


To fine-tune your custom LLM model, follow these steps:

1.  Convert your LLM checkpoint into the ``Huggingface`` format and save it to ``./pretrained-models/FOLDER_OF_YOUR_LLM``.
#.  Execute the following command

    .. code-block:: console

        (pandallm) $ python train.py --model llama-7b --ckpt_path ${FOLDER_OF_YOUR_LLM}

    This command will initiate the fine-tuning process using the ``llama-7b`` model and the checkpoint from your specified directory (``./pretrained-models/FOLDER_OF_YOUR_LLM``).



Optimization
------------

General settings
^^^^^^^^^^^^^^^^

The PandaLLM framework provides several features for training, including automatic gradient accumulation, `NVLAMB <https://arxiv.org/abs/1904.00962>`_ optimizer integration, and quantization-aware training based on `BitsandBytes <https://github.com/facebookresearch/bitsandbytes>`_. To customize the training hyperparameters, you can specify the following arguments. Here is a description of each argument:


--per_gpu_train_batch_size  The batch size for each GPU during training. The default value is :math:`1`.

--per_gpu_eval_batch_size  The batch size for each GPU during evaluation. The default value is :math:`2`.

--optimizer  The training optimizer. The default value is ``"AdamW"``.

--learning_rate  The learning rate for each batch of the model during training. The default value is :math:`0.001`.

--lr_scheduler  The learning rate scheduler options, including ``"linear"``, ``"cosine"``, ``"constant"``, ``"poly"``, and ``"warmup"``. The default value is ``"warmup"`` when the argument is not specified.

--gradient_accumulation_steps  Number of gradient accumulation steps before performing a backward/update pass. The default value is :math:`64`.

--weight_decay  The weight decay applied to all parameters of the model. The default value is :math:`0.00`.

--adam_epsilon  :math:`\varepsilon` value for the Adam optimizer. The default value is :math:`10^{-6}`.

--adam_betas  :math:`\beta` coefficients used for computing moving averages of gradients and squared gradients in the Adam optimizer. The default value is :math:`(0.9, 0.99)`.

--max_grad_norm  Maximum norm for gradient clipping. The default value is :math:`0.3`.

--num_train_epochs  The total number of training epochs. The default value is :math:`1`.

--max_steps  The maximum number of training steps. The default value is :math:`-1`, indicating no maximum limit.

--warmup_proportion  Proportion of training steps to perform linear learning rate warmup. The default value is :math:`0`.

--warmup_steps  Number of warmup steps for learning rate warmup. The default value is :math:`50`.

--bit_training  This ``boolean`` argument specifies the bit training mode for quantization-aware training. It determines the precision of weights and activations during training. The default value is ``False``.


To finetune a ``Panda-7B`` model with a learning rate of :math:`0.002` for :math:`2` epochs, execute the following command:

.. code-block:: console

        (pandallm) $ python train.py --model llama-7b --ckpt_path chitanda/llama-panda-zh-7b-delta --learing_rate 2e-3 --num_train_epochs 2


Low-rank adaptation (LoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^

PandaLLM supports `LoRA <https://github.com/huggingface/peft>`_ finetuning for LLMs. For example, to initiate the training process for the ``LlaMA-65B`` model with LoRA, execute the following command:

.. code-block:: console

        (pandallm) $ python train.py --model llama-65b --use_lora --lora_r 64 --lora_alpha 16 --lora_dropout 0.05

You can customize the behavior of LoRA during the training process of LLMs by specifying the following arguments.

--use_lora  This ``boolean`` argument enables the usage of LoRA (Local Relevance Adaptation) during the training process. When specified, LoRA will be incorporated into the training of LLMs.

--lora_r  This argument determines the number of local neighbors considered for each token during LoRA adaptation. The default value is set to :math:`64`.

--lora_alpha  This argument controls the strength of adaptation for LoRA. It influences the extent to which the model adapts to local relevance. The default value is set to :math:`16`.

--lora_dropout  This argument specifies the dropout rate to apply during LoRA adaptation. Dropout helps to regularize the training process and prevent overfitting. The default value is set to :math:`0.05`.


Quantization-aware training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

PandaLLM enables quantization-aware training based on the `BitsandBytes <https://github.com/facebookresearch/bitsandbytes>`_ framework. For example, to train a ``LlaMA-65B`` model using  `BitsandBytes` quantization scheme with :math:`4`-bit precision, execute the following command:

.. code-block:: console

        (pandallm) $ python train.py --model llama-65b --use_quant

