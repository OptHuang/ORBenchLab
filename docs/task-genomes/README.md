# Source-bound candidate task genomes

这两个文件是从 2026-08-27 的 metadata-only intake shortlist 产生的
**candidate genome**，不是可直接运行的 ORWorkbench family。它们故意不放在
`src/orworkbench/families/`，也没有 `hooks`、verifier 实现或可打包的 task
package；必须经过人工阅读原文、许可证核验、实例作者化和独立 verifier
实现后，才能转成正式任务。

## 共同验收边界

每个候选都要求：

1. `oracle` 能生成一个由独立 verifier 重新检查的可行解；
2. `nop` 和逐条 mutation control 被 verifier 正确拒绝；
3. 至少 3 个 seed、3 个难度 level、2 个独立 agent system，且每个格子
   使用相同 seed 集；
4. 记录 `verifier_digest`、`instance_digest`、`source_intake_item_uid` 和
   `source_content_digest`，才能进入 screening；
5. 任何“提示能恢复某个卡点”的结论都必须使用同一 checkpoint 的 L0
   null arm 和重复 continuation。当前 Harbor 0.16.1 只支持观察/重启边界，
   因而本文件把干预标为 `restart-with-hint`，不宣称已有 mid-episode 注入；
6. `flat`、`non-monotone` 或 `underpowered` 的轴保留为结果，不得被改写成
   “难度轴”。

## 选择依据与限制

选择依据只有 intake 中的标题、摘要、URL 和 digest：

* `multi-agent-vrp-recovery.yaml` 对应一个公开的 VRP 多智能体环境论文，
  适合把事件重规划、冻结操作和因果变更做成可验证约束；
* `cir-constraint-audit.yaml` 对应一个强调 canonical intermediate
  representation 与复合运营规则的论文，适合直接针对本次 IndustryOR_12
  中“自检声称通过、独立 verifier 发现 coupled constraint 违例”的失败模式。

这不是对论文实验结果、代码许可证或数据集可用性的确认。正式 authoring
前必须从原文/仓库建立 source packet，并把以下 `metadata_only` 限制改成
已核验的来源记录；不会把摘要文本自动放进 agent prompt。

