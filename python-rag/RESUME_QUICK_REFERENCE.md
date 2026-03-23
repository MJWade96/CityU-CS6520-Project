# 🔄 断点续传功能 - 快速参考

## 📌 一句话总结

**长时间运行的评估任务异常中断后，可以自动从上次中断的位置继续，无需从头开始。**

---

## 🚀 快速开始

### 方法 1：直接运行（最常用）

```bash
# 完整评估
python complete_eval.py

# 增强版评估
python enhanced_eval.py
```

**系统会自动**：
- ✅ 检测是否有中断的任务
- ✅ 如果有，自动从上次中断位置继续
- ✅ 如果没有，从头开始正常运行
- ✅ 完成后自动清理检查点

### 方法 2：智能启动脚本

```bash
# 自动检测并选择恢复哪个任务
python run_with_resume.py --auto

# 运行特定脚本（自动恢复）
python run_with_resume.py complete_eval

# 强制从头开始（不恢复）
python run_with_resume.py complete_eval --no-resume
```

---

## 💡 常见场景

### 场景 1：网络中断

```
第一次运行 → 第 150 题时 API 超时 → 进度自动保存
修复网络 → 再次运行 → 自动从第 151 题继续 ✅
```

### 场景 2：手动中断

```
运行中 → 按 Ctrl+C → 进度自动保存
稍后继续 → 再次运行 → 自动恢复进度 ✅
```

### 场景 3：系统崩溃

```
运行中 → 电脑崩溃 → 进度已保存在磁盘
重启后 → 再次运行 → 自动恢复进度 ✅
```

---

## 📊 工作原理

```
启动任务
   ↓
检查检查点 → 有 → 加载进度 → 从断点继续
   ↓          ↓
  没有       运行中...
   ↓          ↓
从头开始    每题保存进度
   ↓          ↓
正常运行    中断？→ 是 → 返回检查步骤
            ↓
           否 → 完成 → 清除检查点
```

---

## 📁 检查点文件

**位置**：`results/evaluation/`

**文件**：
- `checkpoint_complete_eval.json` - 完整评估检查点
- `checkpoint_enhanced_eval.json` - 增强版检查点
- `checkpoint.backup.json` - 备份检查点

**内容**：
- ✅ 当前进度（已处理题目数/总题目数）
- ✅ 详细结果（每道题的答案和评估）
- ✅ 配置信息（top-k、模型等）
- ✅ 时间戳和耗时

---

## 🛠️ 高级操作

### 查看中断状态

```bash
python run_with_resume.py --auto
```

输出示例：
```
📋 Found 1 interrupted evaluation(s):

1. complete_eval
   Dataset: Test Set
   Progress: 150/500
   Timestamp: 2026-03-22T14:30:45
```

### 手动清理检查点

```bash
# 方法 1：使用参数
python run_with_resume.py complete_eval --no-resume

# 方法 2：手动删除
rm results/evaluation/checkpoint_*.json
```

### 查看检查点详情

```python
import json
with open('results/evaluation/checkpoint_complete_eval.json', 'r') as f:
    data = json.load(f)
    print(f"Progress: {data['processed_questions']}/{data['total_questions']}")
    print(f"Accuracy: {data['correct_count']}/{data['total_count']}")
```

---

## ⚡ 性能影响

| 操作 | 耗时 | 影响 |
|------|------|------|
| 保存检查点（每题） | 5-10ms | <0.1% |
| 加载检查点 | 50-100ms | 一次性 |
| 恢复流程 | 1-2 秒 | 一次性 |

**结论**：影响极小，可忽略不计 ✅

---

## 🎯 收益计算

假设评估 500 题，每题 10 秒：

| 中断时机 | 无断点续传 | 有断点续传 | 节省 |
|---------|----------|----------|------|
| 100 题后 | 重跑 500 题 (1.4h) | 重跑 400 题 (1.1h) | 0.3h |
| 300 题后 | 重跑 500 题 (1.4h) | 重跑 200 题 (0.6h) | 0.8h |
| 499 题后 | 重跑 500 题 (1.4h) | 重跑 1 题 (10s) | 1.4h |

**最大节省**：1.4 小时 + 500 次 API 调用！

---

## ❓ 常见问题

### Q: 检查点会占用很多空间吗？
A: 每个检查点几 MB，任务完成后自动删除，无需担心。

### Q: 如果检查点损坏怎么办？
A: 系统有备份机制，会自动尝试加载备份。如果都损坏，会从头开始。

### Q: 可以同时运行多个相同任务吗？
A: 不建议，可能导致检查点冲突。

### Q: 如何确认恢复成功？
A: 启动时会显示：
```
⚠️  Found interrupted evaluation: 150/500
🔄 Resuming test set evaluation from question 151
```

### Q: 恢复后会重复处理最后一题吗？
A: 可能会，但不影响结果准确性，系统会覆盖之前的结果。

---

## 📝 最佳实践

1. ✅ **直接运行即可** - 系统自动处理一切
2. ✅ **保持稳定网络** - 减少中断概率
3. ✅ **监控系统资源** - 避免内存不足
4. ✅ **使用开发集测试** - 先用小数据测试
5. ✅ **保留日志** - 便于排查问题

---

## 🆘 故障排除

### 问题 1：无法加载检查点
```
Warning: Could not load checkpoint
```
**解决**：删除损坏的检查点，从头开始
```bash
rm results/evaluation/checkpoint_*.json
```

### 问题 2：进度没有保存
**可能原因**：
- 目录没有写权限
- 磁盘空间不足
- 第一题前就崩溃了

**解决**：
```bash
# 检查权限
ls -la results/evaluation/

# 检查空间
df -h

# 手动创建目录
mkdir -p results/evaluation
```

### 问题 3：恢复后重复处理
**正常现象**，最后一题可能被重新处理，结果会覆盖。

---

## 📚 详细文档

完整使用指南：[`RESUME_GUIDE.md`](file://./RESUME_GUIDE.md)  
技术实现细节：[`断点续传实现总结.md`](file://./断点续传实现总结.md)

---

## 🎉 总结

**断点续传功能让你**：
- ✅ 不用担心网络中断
- ✅ 不用担心系统崩溃
- ✅ 不用担心手动误操作
- ✅ 节省时间和 API 配额

**使用方法**：
```bash
# 就像平常一样运行
python complete_eval.py
```

**剩下的交给系统**！🚀

---

**更新日期**：2026-03-22  
**版本**：v1.0
