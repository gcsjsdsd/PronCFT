# PronCFT

**Pronunciation Contextual Fine-Tuning for Contextual Biasing of Out-of-Vocabulary Words in LLM-Based ASR**

[论文 · arXiv:2610.25766](https://arxiv.org/abs/2610.25766) · [PDF](https://arxiv.org/pdf/2610.25766) · [数据工具](docs/data-tools.md) · [评测与发布范围](docs/evaluation.md)

Yifan Xu, Kaixun Huang, Mengzhe Geng, Mingyu Cui, Shen Huang, Shidong Shang, Xunying Liu

The Chinese University of Hong Kong · Tenlai Team

本仓库介绍 **PronCFT 论文**，并提供与论文相关的**测试数据错拼变体构造与拼写召回评分工具**。当前发布的是部分数据工具；完整测试集、模型权重及训练／推理实现尚未包含在仓库中。

## 研究问题

基于大语言模型的自动语音识别（LLM-based ASR）容易把罕见词和词表外词（OOV）识别为发音相近的常见词。仅提供候选词及其音素（表示发音的符号序列），也不保证模型会将候选发音与语音内容对应起来：模型可能只把音素标注的存在当作选择候选词的提示。

## 方法概览

PronCFT 训练模型建立语音、候选发音与输出拼写之间的联系：

1. **结构化监督**：先输出语句的音素，再输出与候选词的匹配结果，最后生成转写文本。
2. **拼写扰动**：改变候选词拼写，保留其已存储的音素标签，并同步对应输出，减少对熟悉拼写的依赖。
3. **逐阶段移除中间输出**：通过课程训练逐步去掉辅助输出，得到 PronCFT-C，以降低解码开销和中间错误的传播。

论文基于 Qwen3-ASR-1.7B，冻结音频编码器与适配器，只微调大语言模型。完整方法和实验条件以[论文](https://arxiv.org/abs/2610.25766)为准。

## 主要结果

下表摘录论文 Table 2 的各系统最佳报告设置及 Table 3 的平均词错误率（WER，衡量转写错误的比例；越低越好）。不同系统的最佳输出设置不同，详见论文。

| 系统 | OOV 召回率 ↑ | 通用测试集平均 WER ↓ |
|---|---:|---:|
| Giga-SFT | 59.89% | 6.05% |
| PronCFT | 74.12% | 6.02% |
| PronCFT-C | **83.57%** | **6.01%** |

OOV 评测集包含从公开语音基准筛选的 10,533 句；PronCFT-C 相比 Giga-SFT 的召回率提高 **23.68 个百分点**。论文还使用错拼干扰和错误音素对照，检查模型是否依赖候选发音内容。

**这些是论文报告的实验结果，不是运行本仓库人工示例得到的结果。**

## 已发布的数据工具

| 文件 | 功能 |
|---|---|
| [make_misspell_phonectx.py](make_misspell_phonectx.py) | 构造普通转写的错拼变体，或只扰动候选列表的对照数据 |
| [build_misspell_cot.py](build_misspell_cot.py) | 构造带 `<phoneme>/<match>` 标注的错拼变体，并同步标签 |
| [compute_misspell_recall.py](compute_misspell_recall.py) | 统计替换拼写、原拼写、双命中、未命中与缺失预测 |
| [misspell_rules.py](misspell_rules.py) / [misspell_io.py](misspell_io.py) | 拼写编辑规则、词表过滤、输入输出与文本替换 |
| [examples/](examples/) / [test_misspell.py](test_misspell.py) | 人工输入示例与单元测试 |

这些工具接收调用方准备的转写、候选词、音素和目标标签，不负责从原始语音基准筛选 OOV 样本或生成发音。它们保持输入音素标签，不验证新拼写按自然英语规则是否同音。

## 快速开始

仅使用 Python 标准库，建议 Python 3.8 或以上。以下命令在仓库根目录执行，输出文件须尚不存在。

```bash
# 构造错拼变体，同时更新转写与目标标签
python3 make_misspell_phonectx.py --in examples/plain.jsonl --out plain_misspell.jsonl --vocab examples/vocab.txt --prob 0.5 --seed 42

# 只改目标候选的拼写，保留原始参考转写
python3 make_misspell_phonectx.py --in examples/plain.jsonl --out context_misspell.jsonl --vocab examples/vocab.txt --targets-only --context-only --seed 42

# 运行人工样本测试
python3 -B -m unittest -v test_misspell
```

`examples/vocab.txt` 是很小的演示词表。正式构造数据时应使用自己的完整词表；词表排除检查不等于证明拼写从未出现在模型训练中。

输入格式、结构化变体、预测评分和完整参数见[数据工具使用说明](docs/data-tools.md)。论文评测设计与尚未发布的部分见[评测与发布范围](docs/evaluation.md)。

## 引用

如使用本文方法或相关工具，请引用：

```bibtex
@misc{xu2026proncft,
  title={PronCFT: Pronunciation Contextual Fine-Tuning for Contextual Biasing of Out-of-Vocabulary Words in LLM-Based ASR},
  author={Yifan Xu and Kaixun Huang and Mengzhe Geng and Mingyu Cui and Shen Huang and Shidong Shang and Xunying Liu},
  year={2026},
  eprint={2610.25766},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2610.25766}
}
```

另提供 [CITATION.cff](CITATION.cff)，便于引用论文。

## 代码与数据授权

部分脚本保留原有版权声明；仓库当前尚未提供代码许可证。第三方语音数据和词表应分别遵守其原始许可。人工示例不包含论文完整测试集。

