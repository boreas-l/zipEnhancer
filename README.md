# ZipEnhancer
- 【来源】该项目来源于阿里通义实验室开源的语音降噪模型ZipEnhancer >>> [项目连接](https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base/summary)  
- 【背景】最近因为项目问题，打算调研测试一下最新的通话语音降噪算法，看了一下，发现阿里开源的ZipEnhancer模型效果好像还不错，打算试一试。 因为modelscope封装比较复杂，调用链路比较深，为了方便修改与测试，基于pytorch，将ZipEnhancer模型的代码逻辑剥离了出来。
- 【记录】
    - 2025.3.4：简单把推理部分提取了出来，并完成测试，基于官方测试用例的去噪效果还不错。
    - 接下来计划更新onnx导出部分，并完善训练部分代码。【官方目前还没提供训练部分代码】
