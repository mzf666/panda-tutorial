Train your LLM
==============

PandaLLM enables efficient training of various LLMs by leveraging the ``DeepSpeed`` acceleration framework. You can train your LLM with a customized configuration using the following command:

.. code-block:: console

    (.venv) $ bash train.sh --model llama-7b

When you execute the train.sh script, it automatically generates a training configuration file at `./conf/tmp.yaml` based on the configuration template file located at `./conf/template.yaml`. Subsequently, the script initiates the training process by executing `./trainer_torch_fsdp_wandb.py`. If you prefer to train your model with a personalized configuration, you can execute the following command:

.. code-block:: console

    (.venv) $ bash train.sh --conf_path ${PATH_TO_YOUR_CONF_FILE}

In the forthcoming sections, we provide a comprehensive overview of the workflow involved in training an LLM using the `train.sh` script.



.. _data_preparation:

Data Preparation
----------------

The first step is to prepare the training datasets for PandaLLM. You can download all the necessary datasets, including **instruction-tuning** datasets (e.g., Wiki-zh) and **pretraining datasets** (e.g., Wudao), `here <https://entuedu-my.sharepoint.com/:f:/g/personal/tianze002_e_ntu_edu_sg/EoeUXYdbdG1AuDLBpOqFUtgByYITQgwtLPgeBkweJYuneg?e=t9583n>`_. Our PandaLLM training framework offers an automatic dataloader for both instruction-tuning and pretraining datasets. To ensure compatibility, please save all the training datasets under the `./dataset` directory in either the instruction-tuning or pretraining dataset format.

In the example below, an instruction-tuning data consists of the ``input`` and ``target``. Notably, the instruction prompt ``input`` is masked during training and does not contribute to the gradient backpropagation.

.. code-block:: json

    {"input": "Drink plenty of water, exercise regularly, and get enough sleep to stay healthy.", "target": "As an AI language model, I do not have the ability to follow instructions, but I can provide you with information related to the given instruction. Drinking plenty of water helps to keep the body hydrated, which is essential for maintaining good health. Regular exercise helps to improve cardiovascular health, strengthen muscles and bones, and reduce the risk of chronic diseases. Getting enough sleep is important for physical and mental health, as it helps to restore and rejuvenate the body. Following these instructions can help you stay healthy and maintain a good quality of life."}

In the example below, an pretraining data consists of ``title`` and ``content``. During training,


.. code-block:: json

    {"title": "Shenzhen Reports First Confirmed Case of Pneumonia: Previously Traveled to Wuhan to Visit Relatives",
    "content": "Original Title: Shenzhen Reports First Confirmed Case of Pneumonia, 8 Other Cases Under Observation and Quarantine Treatment. Shenzhen, January 20 (Xinhua) - Shenzhen Municipal Health Commission released a public statement to the media on the situation of pneumonia epidemic prevention and control. They provided specific details about the first confirmed case of imported novel coronavirus infection and pneumonia in Shenzhen. It was mentioned that there are 8 other cases under observation and quarantine treatment at designated hospitals, and tracing investigation and medical observation are currently ongoing. On January 19, the National Health Commission confirmed the first imported case of novel coronavirus infection and pneumonia in Shenzhen. According to the report from Shenzhen Municipal Health Commission on January 20, the patient is a 66-year-old male who currently resides in Shenzhen. He visited Wuhan to visit relatives on December 29, 2019. On January 3, 2020, he developed symptoms such as fever and fatigue. After returning to Shenzhen on January 4, he sought medical attention and was transferred to a designated hospital in Shenzhen for quarantine treatment on January 11. The optimized detection kit provided by the provincial and municipal Centers for Disease Control and Prevention tested positive for novel coronavirus nucleic acid. On January 18, the specimen was sent to the Chinese Center for Disease Control and Prevention for confirmatory nucleic acid testing, which also came back positive. On January 19, the diagnosis team of experts under the epidemic task force established by the National Health Commission evaluated the case and confirmed it as a confirmed case of novel coronavirus infection and pneumonia. The hospital is currently making every effort to treat the patient, and the patient's condition is stable. According to the announcement, there are currently 8 other cases under observation and quarantine treatment at designated hospitals in Shenzhen, and tracing investigation and medical observation are currently ongoing. Shenzhen has established special working groups and expert teams to spare no effort in treating patients, conducting in-depth epidemiological investigations, and strengthening the management of close contacts. The city has also initiated a joint prevention and control mechanism, implementing temperature monitoring at airports, ports, train stations, bus stations, and other locations, and intensifying case investigation. Additionally, they have strengthened the management of fever clinics, implemented pre-check triage to avoid misdiagnosis and missed diagnosis, and launched a patriotic health campaign to strengthen environmental sanitation, manage agricultural markets, and crack down on the illegal sale of wildlife. Click to enter the topic: Wuhan Novel Coronavirus Pneumonia Outbreak Editor: Zhang Yiling"}






.. _train_models:

Models
------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']


Optimization Settings
---------------------



