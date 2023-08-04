Inference
======

In this project, we support several approaches for inference:

- HuggingFace Transformers' naive model parallel
- Deepspeed inference (tensor parallel)
- Tensor parallel (supported by ``tensor-parallel`` pypi package)

HuggingFace Transformers' Model Parallel
----------------------------------------

This feature can be quite easy to be enabled by specify ``device_map`` during calling ``xxx.from_pretrained`` method.
Note that you this is not distributed evaluation, so you cannot launch multiple processes when using this feature.

Deepspeed Inference
-------------------

Enabling this feature should use a different entrance script `ds_inference.py`. You can launch the process by:

.. code-block:: bash

    deepspeed --include localhost:0,1,2,3 ds_inference.py -cp <config path> -cn <config file name>

Note that this will launch multiple processes at the same time, however, this is still not distributed evaluation, since all processes
should load exactly the same data (tensor parallel). To this end, we have explicitly set `ddp_eval=True` at the entrance script.

Besides, since tensor parallel requires huge communication across different processes, it is not recommended to use this feature across different nodes.
For single node inference, you may need to try if this is faster than naive model parallel, which depends on the bandwidth of your machine.

Tensor Parallel
---------------
