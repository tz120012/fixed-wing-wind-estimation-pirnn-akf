#!/usr/bin/env bash
# 串联跑 Exp-C → Exp-A → Exp-B 三个超参对照实验
# 每个实验独立 log + 独立 run_tag，patience=20 自动早停
#
# Exp-C: lambda_anti_collapse=1.0 + lambda_dir=0.5  （组合最优）
# Exp-A: lambda_anti_collapse=1.0                   （仅释放防崩塌）
# Exp-B: lambda_dir=0.5                              （仅强化方向监督）
#
# 用法（在项目根目录）:
#   nohup bash scripts/run_exp_chain_CAB.sh > logs/exp_chain_CAB.log 2>&1 &
#   disown

set -u  # 未定义变量报错；不用 -e，单实验失败不阻塞下一个

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs

CHAIN_START=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="logs/exp_chain_CAB_summary_${CHAIN_START}.md"

cat > "$SUMMARY_FILE" <<EOF
# Exp-C / Exp-A / Exp-B 串联实验日志

启动时间: $(date '+%F %T')
PID: $$

| 实验 | config | 关键改动 | 启动时间 | 完成时间 | 状态 | 日志 |
|---|---|---|---|---|---|---|
EOF

run_one_exp () {
    local exp_id="$1"      # C / A / B
    local cfg="$2"
    local desc="$3"
    local stamp; stamp=$(date +%Y%m%d_%H%M%S)
    local logf="logs/exp_${exp_id}_${stamp}.log"

    echo
    echo "######################################################################"
    echo "# 启动 Exp-${exp_id}: ${desc}"
    echo "# config = ${cfg}"
    echo "# log    = ${logf}"
    echo "# time   = $(date '+%F %T')"
    echo "######################################################################"

    echo "| Exp-${exp_id} | ${cfg} | ${desc} | $(date '+%F %T') | — | running | ${logf} |" >> "$SUMMARY_FILE"

    # 实际训练
    python -u src/3_train_pigru.py --config_path "$cfg" > "$logf" 2>&1
    local rc=$?

    local now; now=$(date '+%F %T')
    if [ "$rc" -eq 0 ]; then
        echo "✅ Exp-${exp_id} 正常退出 (rc=0)，结束时间 ${now}"
        # 在 summary 表里把最后一行的 status 改为 ok
        sed -i "s|| running | ${logf} ||| ok    | ${logf} ||" "$SUMMARY_FILE"
    else
        echo "❌ Exp-${exp_id} 异常退出 (rc=${rc})，结束时间 ${now}"
        sed -i "s|| running | ${logf} ||| FAIL  | ${logf} ||" "$SUMMARY_FILE"
    fi

    # 抽取该实验的 best 指标摘要
    {
        echo
        echo "## Exp-${exp_id} 摘要 (rc=${rc})"
        echo
        echo '```'
        grep -E "lambda_anti_collapse|lambda_dir =|开始第|🎯 single|early_stopping_patience|✅ 保存主最佳模型|❌ 验证组合分数 未改善|早停触发|训练完成|最佳验证RMSE|最佳风速大小|风向误差|MAE=" "$logf" 2>/dev/null | tail -30
        echo '```'
    } >> "$SUMMARY_FILE"
}

run_one_exp C config/config_strat_expC.yaml "ac=1.0 + dir=0.5"
run_one_exp A config/config_strat_expA.yaml "ac=1.0 (dir 维持 0.1)"
run_one_exp B config/config_strat_expB.yaml "dir=0.5 (ac 维持 5.0)"

echo
echo "====================================================================="
echo "三实验串联结束: $(date '+%F %T')"
echo "汇总报告: $SUMMARY_FILE"
echo "====================================================================="
