Quick Start
===========

.. autosummary::
   :toctree: generated

`PandaLLM Github <https://github.com/dandelionsllm/pandallm>`_

We will take deployment and training of Panda-13B as an example.

.. _installation:

Installation
------------

1. Download our code from github

.. code-block:: console

    $ git clone https://github.com/dandelionsllm/pandallm

2. Install the requirements in a new environment

.. code-block:: console

    $ conda create -n pandallm python=3.10
    $ conda activate pandallm
    (pandallm) $ pip install -r requirements.txt
    (pandallm) $ mkdir pretrained_model


.. _quick_deploy:

Quick Deployment
----------------


1. Download ``LlaMA-13B`` from `Huggingface <https://huggingface.co/huggyllama/llama-13b>`_.

2. Download our model form Huggingface. Since the model file is too large for git clone, you may manually download the model files from `here <https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta>`_.

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

3. Move the downloaded files to the corresponding directory.

.. code-block:: console

     (pandallm) $ cd ..
     (pandallm) $ mv delta-models/ ./

4. Convert ``"delta-model"`` to a pretrained model. Replace ${PATH_TO_YOUR_MODEL} with your desired model path, where your model will be saved there.

.. code-block:: console

    (pandallm) $ python apply_delta.py --base_model ${PATH_TO_YOUR_MODEL} --target_model ./pretrained_model/panda-13B --delta_model ./delta-models/llama-panda-13b-zh-wudao-chat-delta/checkpoint-3000-delta

5. Run the following command to deploy the chatbot.

.. code-block:: console

    (pandallm) $ python run_chat.py --model_path ./pretrained_model/panda-13B --query "write a peom"


.. _quick_train:

Quick Train
-----------

Before you can directly train the model with the following commands, make sure you have finish the :ref:`installation <installation>`.

1. Prepare the training data. You can download the training data from `here <https://entuedu-my.sharepoint.com/:f:/r/personal/tianze002_e_ntu_edu_sg/Documents/Panda%E5%A4%A7%E6%A8%A1%E5%9E%8B/dataset?csf=1&web=1&e=0i1Oiu>`_. Please put the data folders at ``./dataset``.


2. Run the following command to train the model:

.. code-block:: console

  (pandallm) $ PAD_TOKEN="</s>" deepspeed --include localhost:0,1,2,3,4,5,6,7  trainer_base_ds_mul.py -cp conf/llama/zh/ -cn llama_13b_zh_instruct_sft_combine_v1_0_ds

If you have less than :math:`8` GPUs, you can change the ``--include parameter`` to the GPUs you have, e.g. ``"--include localhost:0,1,2,3"`` if you have :math:`4` GPUS on one server.