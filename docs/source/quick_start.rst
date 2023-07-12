Quick Start
==============


### 准备数据


将数据放入 `dataset` 文件夹中
 
数据格式样例：


	[{"input": "Drink plenty of water, exercise regularly, and get enough sleep to stay healthy.", "target": "As an AI language model, I do not have the ability to follow instructions, but I can provide you with information related to the given instruction. Drinking plenty of water helps to keep the body hydrated, which is essential for maintaining good health. Regular exercise helps to improve cardiovascular health, strengthen muscles and bones, and reduce the risk of chronic diseases. Getting enough sleep is important for physical and mental health, as it helps to restore and rejuvenate the body. Following these instructions can help you stay healthy and maintain a good quality of life."}, {"input": "Eat a balanced diet, avoid smoking and excessive alcohol consumption, and maintain a healthy weight to stay healthy.", "target": "As an AI language model, I don't have the ability to follow instructions or eat. But, I understand that eating a balanced diet, avoiding smoking and excessive alcohol consumption, and maintaining a healthy weight are important for staying healthy. These lifestyle changes can help prevent chronic diseases such as heart disease, diabetes, and certain types of cancer. So, it's important to make healthy choices and take care of our bodies."}]


此样例中包含两条数据，每条数据包含`input`和`target`部分。其中`input`部分为intruction tuning的prompt，此部分在训练中会被mask掉，因此不会回传loss。`target`部分为训练部分，模型需要根据`input`来生成对应的`target`。
对于pretraining的数据，我们将所有数据都放在`title`与`content`里面，如下。


	[{
        "id": 7,
        "uniqueKey": "fdcadef664635a5a757d4d529efc218a",
        "titleUkey": "cc470d358425a945fab97ef229cd8090",
        "dataType": "科技",
        "title": "深圳通报首例肺炎确诊病例情：曾赴武汉探亲",
        "content": "原标题：深圳通报首例肺炎确诊病例情况另有8例观察病例隔离治疗 新华社深圳1月20日电（记者白瑜）深圳市卫健委20日向媒体公开发布肺炎疫情防控工作情况通报，介绍深圳首例输入性新型冠状病毒感染肺炎确诊病例的具体情况，并称另有8例观察病例在定点医院隔离治疗，追踪调查和医学观察正在进行中。 1月19日，国家卫生健康委确认深圳首例输入性新型冠状病毒感染的肺炎确诊病例。深圳市卫健委20日通报称，患者为男性，66岁，现居深圳，2019年12月29日赴武汉探亲，2020年1月3日出现发热、乏力等症状，1月4日返深后就诊，1月11日转至深圳市定点医院隔离治疗。经省、市疾控中心采用优化后的检测试剂盒检测，呈新型冠状病毒核酸阳性。1月18日，标本送至中国疾控中心进行病毒核酸复核检测，结果为阳性。1月19日，经国家卫生健康委疫情领导小组下设的诊断组专家对该病例进行评估，确认为新型冠状病毒感染的肺炎确诊病例。现医院组织全力救治，患者病情稳定。通报称，目前深圳另有8例观察病例在定点医院隔离治疗，追踪调查和医学观察正在进行中。 深圳成立了专项工作组和专家小组，全力救治患者，深入开展流行病学调查，加强密切接触者跟踪管理；启动了全市的联防联控工作机制，在机场、码头、火车站、客运站等场所启动体温监测工作，加强病例排查；并且加强发热门诊管理，做好预检分诊，避免漏诊和误诊；同时开展爱国卫生运动，加强环境整治，做好农贸市场管理，取缔违法售卖野生动物行为。点击进入专题：武汉发生新型冠状病毒肺炎责任编辑：张义凌"
    }]


数据集下载地址：




## 设置超参数


我们提供了一套预设的超参数，包含7B和13B的训练参数。理论上来说用户可以直接使用我们的超参数来得到一个跟我们release的效果一样的模型。用户也可以根据自己的喜好和机器容量来自由修改。


7B的参数文件：


	llama_7b_zh_instruct_sft_combine_v1_0_ds.yaml


13B的参数文件：


	llama_13b_zh_instruct_sft_combine_v1_0_ds.yaml


### 设置参数的经验








## 如何运行


	deepspeed --include localhost:0,1,2,3,4,5,6,7  trainer_base_ds_mul.py -cp conf/llama/zh/ -cn llama_7b_zh_instruct_sft_combine_v1_0_ds



