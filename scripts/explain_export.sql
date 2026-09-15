-- @author: ztwz
-- 两阶段导出的 EXPLAIN 性能验证（线上 vibee 库 localhost:3307 执行）
-- 注意：EXPLAIN 只看计划不执行，安全；EXPLAIN ANALYZE 会真实执行一次。

-- 0. 确认 uniq_task_uuid 索引存在（阶段1 依赖其覆盖扫描）
SHOW INDEX FROM manage_model_log;

-- 1. 阶段1 执行计划：覆盖索引取全部目标 id
--    预期 key=uniq_task_uuid，Extra 含 "Using index"（覆盖，零回表），无 filesort（id 排序在应用侧）
EXPLAIN
SELECT id FROM manage_model_log FORCE INDEX (uniq_task_uuid)
WHERE task = 'user_chat_analysis' AND id > 0;

-- 2. 阶段1 实测耗时（真实执行，返回 ~8.7 万个 id，纯索引扫描）
--    预期 < 1 秒；若到秒级以上说明线上该索引页不在内存，属于一次性预热成本
EXPLAIN ANALYZE
SELECT id FROM manage_model_log FORCE INDEX (uniq_task_uuid)
WHERE task = 'user_chat_analysis' AND id > 0;

-- 3. 阶段2 执行计划：主键 IN 点查（示例用 10 个 id，实际批次为 1000 个）
--    预期 type=range, key=PRIMARY，rows≈IN 个数，无 filesort
--    把下面的 id 换成阶段1 查到的任意 10 个真实 id
EXPLAIN
SELECT id, prompt, response FROM manage_model_log
WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  AND status = 1 AND prompt != '' AND response NOT IN ('', '{}');

-- 4. 阶段2 实测耗时（10 个 id 的点查样本）
--    外推到 1000 个/批 × 87 批 = 总点查 8.7 万次，对比旧方案 245 万次回表
EXPLAIN ANALYZE
SELECT id, prompt, response FROM manage_model_log
WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  AND status = 1 AND prompt != '' AND response NOT IN ('', '{}');

-- 结果判读：
-- 段1: key=uniq_task_uuid + Extra 含 Using index   → 覆盖扫描生效，零回表
-- 段2: actual time 毫秒~亚秒级                    → 阶段1 达标
-- 段3: type=range + key=PRIMARY + 无 filesort      → 点查计划正确
-- 段4: 单批 1000 个 id 点查预计 < 1 秒             → 全量 87 批 ≈ 1-2 分钟
