# LaBraM portable equal-probability ensemble v29 冻结协议

**状态：** v28 public OOF 后形成的 public-adaptive exploratory 候选；private direct-token 特征生成和标签读取前冻结。  
**禁止表述：** preregistered public confirmation、fresh external validation、private-selected ensemble。

## 固定模型

```text
arm H:
  frozen LaBraM block-9 H -> fold-local H-only 16-parameter reasoner

arm D:
  frozen LaBraM block-9 five phase token components
  -> fold-local 206-parameter rank-1 direct-token reasoner

within each fold:
  p_equal = 0.5 * p_H + 0.5 * p_D

deployment:
  p_final = mean of five fold-specific p_equal
```

- 不学习 ensemble weight，不使用 confidence gate，不扫描概率/排名/logit融合方式。
- 不使用跨域 gate 已失败的 fine feature family。
- public 评价使用每名患者所属 outer fold 的两个 held-out prediction，单位为患者。
- private 先对五个 frozen fold 产生 target-blind prediction，再等权平均；标签只在 prediction artifact 发布后打开。
- private 已历史开标，结果只能称 post-open exploratory clinical concordance。

## Public 启动门

相对于 H-only 和 v17，必须同时满足：

1. strict Top-1 不低于 v17 的 50%；
2. neighborhood-4 严格高于 DeepSOZ 论文中心值 0.744；
3. macro AP 严格高于 v17 的 0.5412731；
4. far error 不高于 v17 的 24；
5. 至少四折 strict 不低于 H-only；
6. 输出 finite，固定 C18，PZ masked。

通过只授权一次 target-blind private inference；不授权根据 private 结果改权重、阈值、seed、fold 或候选空间。

