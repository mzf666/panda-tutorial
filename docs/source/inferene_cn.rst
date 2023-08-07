推理
======

在此项目中，我们支持几种推理方法：

- HuggingFace Transformers的初级模型并行
- Deepspeed推理（张量并行）
- 张量并行（由``tensor-parallel`` pypi包支持）

HuggingFace Transformers的模型并行
----------------------------------------

通过在调用``xxx.from_pretrained``方法期间指定``device_map``，可以很容易地启用此功能。
请注意，这不是分布式评估，因此在使用此功能时，您不能启动多个进程。

Deepspeed 推理
-------------------

启用此功能应使用不同的入口脚本`ds_inference.py`。您可以通过以下方式启动进程：

.. code-block:: bash

    deepspeed --include localhost:0,1,2,3 ds_inference.py -cp <配置路径> -cn <配置文件名>

请注意，这将同时启动多个进程，但是，这仍然不是分布式评估，因为所有进程
应完全加载相同的数据（张量并行）。为此，我们在入口脚本中明确设置了`ddp_eval=True`。

此外，由于张量并行需要在不同的进程之间进行大量通信，因此不建议在不同的节点之间使用此功能。
对于单节点推理，您可能需要尝试这是否比初级模型并行更快，这取决于您机器的带宽。

张量并行
---------------
