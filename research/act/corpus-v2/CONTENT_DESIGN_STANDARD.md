# Content Design Standard — Understanding Hierarchy

这一版本新增的核心约束是：**知识树层级正确，不代表文章内部的理解层级正确。** 每个页面还必须设计知识进入读者脑中的顺序。

## Model / Architecture Overview Pages

在进入论文级细节前，必须先完成以下认知：

1. 这是什么类型的东西；
2. 它解决什么问题；
3. 输入是什么；
4. 输出是什么；
5. 一张最小完整数据流图；
6. 相比前置方法，最核心的结构变化是什么。

之后才允许展开模块、数学、training/inference 和实现差异。

## Mechanism Pages

先说明机制在完整系统中的职责，再展示一次完整计算，然后进入公式、shape、性质和与上下游机制的连接。

## Mathematical Pages

先明确数学对象在描述什么，再给直觉与正式定义；公式必须说明变量、尺度、为什么出现以及在 AI 模型中怎样被使用。

## Progressive Disclosure

每向下一节，读者脑中的系统只增加一层必要结构。不要在第一次出现模型时同时展开所有 implementation details；不要让读者在还不知道整体模型形状时先学习局部术语。

## Paper Grounding

文章顺序由理解路径决定，事实边界由原论文与官方实现决定。**Paper Grounding 不等于按照 paper section 顺序翻译。**
