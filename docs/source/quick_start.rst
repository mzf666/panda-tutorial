Quick Start
===========

.. autosummary::
   :toctree: generated

Panda github https://github.com/dandelionsllm/pandallm

Installation
------------

1. Download our code from github

.. code-block::

    git clone https://github.com/dandelionsllm/pandallm

2. Install the requirements in a new environment

.. code-block::

    conda create -n pandallm python=3.10
    conda activate pandallm
    pip install -r requirements.txt
    mkdir pretrained_model


Quick Deployment
----------------


1. Download LLaMA-13B from huggingface: https://huggingface.co/huggyllama/llama-13b

2. Download our model form huggingface


.. code-block::

    mkdir delta-models
    cd delta-models
    git clone https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta

Because the model file is too large for git clone, you may manually download the model files from https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta

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

Move the downloaded files to the corresponding directory

.. code-block::

    cd ..
    mv delta-models/ {path_to_pandallm}/

Convert delta-model to the pretrained model

.. code-block::

    python {path_to_pandallm}/apply_delta.py --base_model {path_to_your_LLaMA-13B_model} --target_model {path_to_pandallm}/pretrained_model/panda-13B --delta_model {path_to_pandallm}/delta-models/llama-panda-13b-zh-wudao-chat-delta/checkpoint-3000-delta

3. Run the following command to deploy the chatbot

.. code-block::

    python run_chat.py --model_path {path_to_pandallm}/pretrained_model/panda-13B --query "write a peom"



Quick Train
-----------

If you have already done the 1-3 steps in Quick Deployment, you can directly run the following command to train the model. Otherwise, please follow the 1-3 steps in Quick Deployment.

1. Prepare the training data

You can download the training data from https://entuedu-my.sharepoint.com/:f:/r/personal/tianze002_e_ntu_edu_sg/Documents/Panda%E5%A4%A7%E6%A8%A1%E5%9E%8B/dataset?csf=1&web=1&e=0i1Oiu

After you download the training data, please put the data folderunder the root path of {path_to_pandallm}.


2. Run the following command to train the model.

.. code-block::

    deepspeed --include localhost:0,1,2,3,4,5,6,7  trainer_base_ds_mul.py -cp conf/llama/zh/ -cn llama_13b_zh_instruct_sft_combine_v1_0_ds

If you have less than 8 GPUs, you can change the --include parameter to the GPUs you have, e.g. --include localhost:0,1,2,3 if you have 4 GPUS on one server.