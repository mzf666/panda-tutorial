快速开始
===========

.. autosummary::
   :toctree: generated

`PandaLLM Github <https://github.com/dandelionsllm/pandallm>`_

我们以部署和训练 Panda-13B 为例。

.. _installation:

安装
------------

1. 从Github下载我们的代码

.. code-block:: console

    $ git clone https://github.com/dandelionsllm/pandallm

2. 在新的环境中安装所需的依赖

.. code-block:: console

    $ conda create -n pandallm python=3.10
    $ conda activate pandallm
    (pandallm) $ pip install -r requirements.txt
    (pandallm) $ mkdir pretrained_model

.. _quick_deploy:

快速部署
----------------

1. 从 `Huggingface <https://huggingface.co/huggyllama/llama-13b>`_ 下载 ``LlaMA-13B`` 。

2. 从Huggingface下载我们的模型。由于模型文件对于git clone来说太大了，您可以从 `这里 <https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta>`_ 手动下载模型文件。

.. code-block:: console

    (pandallm) $ mkdir delta-models
    (pandallm) $ cd delta-models
    (pandallm) $ git clone https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta

..
 [and] wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/pytorch_model-00001-of-00006.bin
 wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/pytorch_model-00002-of-00006.bin
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/pytorch_model-00003-of-00006.bin
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/pytorch_model-00004-of-00006.bin
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/pytorch_model-00005-of-00006.bin
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/pytorch_model-00006-of-00006.bin
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/config.json
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/generation_config.json
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/pytorch_model.bin.index.json
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/special_tokens_map.json
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/tokenizer.model
    wget https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta/resolve/main/checkpoint-3000-delta/tokenizer_config.json


3. 将下载的文件移动到相应的目录。

.. code-block:: console

     (pandallm) $ cd ..
     (pandallm) $ mv delta-models/ ./

4. 将 ``"delta-model"`` 转换为预训练的模型。请将 ${PATH_TO_YOUR_MODEL} 替换为您想要保存的模型路径。

.. code-block:: console

    (pandallm) $ python apply_delta.py --base_model ${PATH_TO_YOUR_MODEL} --target_model ./pretrained_model/panda-13B --delta_model ./delta-models/llama-panda-13b-zh-wudao-chat-delta/checkpoint-3000-delta

5. 运行以下命令部署聊天机器人。

.. code-block:: console

    (pandallm) $ python run_chat.py --model_path ./pretrained_model/panda-13B --query "write a peom"


.. _quick_train:

快速训练
-----------

在直接使用以下命令训练模型之前，请确保您已经完成了 :ref:`安装 <installation>`.

1. 准备训练数据。您可以从 `这里 <https://entuedu-my.sharepoint.com/:f:/r/personal/tianze002_e_ntu_edu_sg/Documents/Panda%E5%A4%A7%E6%A8%A1%E5%9E%8B/dataset?csf=1&web=1&e=0i1Oiu>`_ 下载训练数据。请将数据文件夹放在 ``./dataset``。

2. 运行以下命令来训练模型：

.. code-block:: console

  (pandallm) $ PAD_TOKEN="</s>" deepspeed --include localhost:0,1,2,3,4,5,6,7  trainer_base_ds_mul.py -cp conf/llama/zh/ -cn llama_13b_zh_instruct_sft_combine_v1_0_ds

如果您的服务器上少于 :math:`8` 个GPUs，您可以将 ``--include 参数`` 更改为您拥有的GPUs，例如 ``"--include localhost:0,1,2,3"`` 如果您在一个服务器上有 :math:`4` GPUS。
