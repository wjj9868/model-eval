# @author: ztwz
"""多模型 1000 条评测编排：基座 2B/4B 各自起 vLLM 串行评测，smoke/10k 复用已有 dump
（模型输出）但用**当前评分器版本**重新打分，最后汇总渲染对比图并补打观测。

设计（口径与性能取舍）：
- 基座 2B/4B 走 vLLM 后端；生成阶段 vLLM 独占 GPU，打分阶段停掉 vLLM 再用 cuda
  （避免 embedding 与推理抢显存）；
- smoke-2B / lora-10k 的**模型输出（dump）不再重新生成**，但分数文件必须用当前评分器
  重打——评分权重曾有变更（intent 0.5/0.2/0.3 → 0.8/0.1/0.1），旧 scores 与本次基座
  口径不一致，直接复用会破坏对比（plot_eval_compare 要求同一评分器版本）；
- 观测文件（--dump-observations 产物）在重打分阶段一并产出。

用法（训练机 Linux，venv 路径按环境覆盖）：
  python scripts/run_compare_all.py \
      --test-file data/split/test_1k.jsonl --limit 1000 \
      --smoke-dump data/tmp_students.jsonl \
      --lora-dump data/tmp_students_10k.jsonl \
      --out-dir data/compare \
      --vllm-venv /workspace/.venv-vllm \
      --train-venv /workspace/.venv-train \
      --title "Qwen3.5 2B vs 4B vs 蒸馏 评测对比"

已存在目标分数文件时跳过对应模型评测（幂等，可断点续跑）。
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib import request

BASE_URL = "http://127.0.0.1:8000"

# 需补跑的基线模型（label 作产物文件名前缀）
BASE_MODELS = [
    {"label": "base_2b", "name": "Qwen/Qwen3.5-2B", "served": "Qwen3.5-2B"},
    {"label": "base_4b", "name": "Qwen/Qwen3.5-4B", "served": "Qwen3.5-4B"},
]

# 对比图曲线顺序：基础 2B → 基础 4B → smoke → 10k-lora
SERIES_DEF = [
    ("基础 2B", "base_2b_scores.jsonl"),
    ("基础 4B", "base_4b_scores.jsonl"),
    ("蒸馏 smoke", "smoke_scores.jsonl"),
    ("蒸馏 10k-lora", "lora_scores.jsonl"),
]


def sh(cmd: str, timeout: int | None = None) -> bool:
    """执行 shell 命令（阻塞），失败打印输出并返回 False"""
    print(f"\n$ {cmd}", flush=True)
    cp = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        print(cp.stdout[-2000:] if cp.stdout else "")
        print(cp.stderr[-2000:] if cp.stderr else "", file=sys.stderr)
        return False
    return True


def wait_port_free(port: int = 8000, timeout: int = 120) -> None:
    """等待端口释放（vLLM 退出干净）"""
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(3)
    raise SystemExit(f"端口 {port} {timeout}s 内未释放，疑似 vLLM 残留（pkill -f 'vllm serve'）")


def wait_ready(log_path: Path, timeout: int = 600) -> str:
    """轮询 /v1/models 直到服务就绪，返回服务端模型 id"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with request.urlopen(BASE_URL + "/v1/models", timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read())
                    if data.get("data"):
                        return data["data"][0]["id"]
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit(f"vLLM 服务 {timeout}s 未就绪，看日志：{log_path}")


def evaluate_base(cfg: dict, args, dump: Path, scores: Path) -> None:
    """单个基座模型：起 vLLM → 生成 → 停 → 打分（GPU 打分）。

    --manual-server 时服务由用户手动起/停（脚本后台启动在某些 Linux 上不生效）：
    脚本只等待 8000 已有服务 → 生成 → 等待用户手动停服务（可交互，or 直接 score 前探测端口空闲）。
    打分阶段 GPU 打分需要服务已停（显存冲突），故手动模式统一 --device cpu。
    """
    venv_train = args.train_venv
    eval_script = args.eval_script
    log = args.out_dir / f"vllm_{cfg['label']}.log"

    if args.manual_server:
        print(f"\n[{cfg['label']}] 手动模式：请自行启动 vLLM 服务后按回车继续，或我等它就绪…", flush=True)
        try:
            input(f"[{cfg['label']}] 启动服务：{cfg['name']}（无忽略）…")  # 等待用户回车确认已起
        except (EOFError, KeyboardInterrupt):
            pass
        served_id = wait_ready(log)
    else:
        wait_port_free(8000)
        start_cmd = (
            f"source {args.vllm_venv}/bin/activate && "
            "VLLM_USE_FLASHINFER_SAMPLER=0 nohup vllm serve "
            f"{cfg['name']} --served-model-name {cfg['served']} "
            f"--max-model-len {args.max_model_len} --port 8000 --gpu-memory-utilization {args.gpu_mem} "
            f"> {log} 2>&1 &"
        )
        if not sh(start_cmd):
            raise SystemExit(f"vLLM 启动失败：{cfg['label']}，看 {log}")
        served_id = wait_ready(log)

    gen_ok = sh(
        f"{venv_train}/bin/python {eval_script} --backend openai --base-url {BASE_URL} "
        f"--stage generate --served-model-name {served_id} "
        f"--input {args.test_file} --limit {args.limit} "
        f"--max-new-tokens {args.max_new_tokens} --dump-outputs {dump} "
        f"--timeout {args.timeout} --concurrency {args.concurrency}"
    )
    if not gen_ok:
        raise SystemExit(f"生成失败：{cfg['label']}，dump 未产出可打分内容")
    print(f"生成完成：{cfg['label']}（dump={dump}）", flush=True)

    if args.manual_server:
        print(f"[{cfg['label']}] 请手动停止 vLLM 服务（释放显存）后按回车继续打分…", flush=True)
        try:
            input("就绪后回车继续（打分走 CPU）…")
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        sh(f"pkill -f 'vllm serve' || true")
        wait_port_free(8000)

    score_device = "cpu" if args.manual_server else args.device
    if not sh(
        f"{venv_train}/bin/python {eval_script} --backend openai "
        f"--stage score --input {args.test_file} --limit {args.limit} "
        f"--dump-outputs {dump} --out {scores} --device {score_device}"
    ):
        raise SystemExit(f"打分失败：{cfg['label']}（out={scores}）")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多模型 1000 条评测编排（基座 vLLM 串行 + 复用已有分数）")
    p.add_argument("--test-file", default="data/split/test_1k.jsonl")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--gpu-mem", type=float, default=0.92)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    p.add_argument("--smoke-dump", default="data/tmp_students.jsonl",
                   help="smoke-2B 已有模型输出 dump（复用，不重新生成）")
    p.add_argument("--lora-dump", default="data/tmp_students_10k.jsonl",
                   help="10k-lora 已有模型输出 dump（复用，不重新生成）")
    p.add_argument("--out-dir", default="data/compare")
    p.add_argument("--vllm-venv", default="/workspace/.venv-vllm")
    p.add_argument("--train-venv", default="/workspace/.venv-train")
    p.add_argument("--eval-script", default="scripts/run_real_eval.py")
    p.add_argument("--title", default="Qwen3.5 2B vs 4B vs 蒸馏 评测对比")
    p.add_argument("--skip-base", action="store_true", help="跳过基座评测（仅汇总已存在的分数文件）")
    p.add_argument("--base-only", default=None, help="只跑指定基座（如 base_2b），逗号分隔")
    p.add_argument("--manual-server", action="store_true",
                   help="基座 vLLM 由你自己启动/停止（脚本后台启动在部分环境不生效；交互式等回车），打分自动转 CPU")
    return p.parse_args()


def rescore_existing(dump: Path, scores: Path, args, obs_path: Path | None = None) -> bool:
    """用**当前评分器版本**给已有模型输出（dump）重新打分（不重新生成）。

    旧 scores 是旧评分口径（intent 权重变更过），直接复用会与本次基座不可比；
    打分不依赖 GPU 推理，走 --stage score 纯打分解耦。
    obs_path：非 None 时顺带生成低分观测文件（用当前口径，低分判定会随权重变化）。
    """
    if not dump.exists():
        print(f"警告：缺模型输出 dump {dump}，跳过该曲线（compare 图将缺一条）", flush=True)
        return False
    if scores.exists() and (obs_path is None or obs_path.exists()):
        print(f"已存在 {scores}（当前评分器口径），跳过重打", flush=True)
        return True
    print(f"重打 {dump} → {scores}（当前评分器）", flush=True)
    cmd = (
        f"{args.train_venv}/bin/python {args.eval_script} "
        f"--stage score --input {args.test_file} --limit {args.limit} "
        f"--dump-outputs {dump} --out {scores} --device {args.device}"
    )
    if obs_path is not None:
        cmd += f" --dump-observations {obs_path}"
    ok = sh(cmd)
    return ok


def main() -> None:
    args = parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.test_file = Path(args.test_file)

    print("评分口径：intent 0.8/0.1/0.1（当前版本），smoke/10k 将按此口径重打", flush=True)

    # 1) smoke / 10k 复用已有模型输出（dump），用当前评分器重新打分
    #    10k-lora 顺带产出低分观测（新权重下低分判定会变化，观测需同步重打）
    obs_path = args.out_dir / "lora_obs_high_risk.jsonl"
    for key in ("smoke", "lora"):
        dump = Path(getattr(args, f"{key}_dump"))
        scores = args.out_dir / f"{key}_scores.jsonl"
        extra_obs = obs_path if key == "lora" else None
        if not rescore_existing(dump, scores, args, obs_path=extra_obs):
            print(f"跳过 smoke/lora 汇总依赖：{key}", flush=True)

    # 2) 基座 2B/4B 评测（可断点：目标分数存在则跳过）；args Namespace 即全部配置，无需另建 dict
    selected = [m for m in BASE_MODELS if m["label"] in (args.base_only or "").split(",") or not args.base_only]
    if not args.skip_base:
        for cfg in selected:
            dump = args.out_dir / f"{cfg['label']}_dump.jsonl"
            scores = args.out_dir / f"{cfg['label']}_scores.jsonl"
            if scores.exists():
                print(f"已存在 {scores}，跳过 {cfg['label']}", flush=True)
                continue
            evaluate_base(cfg, args, dump, scores)
    else:
        print("--skip-base：跳过基座评测", flush=True)

    # 3) 汇总对比图
    series_args = []
    for name, file in SERIES_DEF:
        path = args.out_dir / file
        if path.exists():
            series_args += ["--series", f"{name}={path}"]
    if len(series_args) < 4:
        print("警告：曲线的分数文件不足 4 条，输出图将缺少部分序列", flush=True)
    plot = sh(
        f"{args.train_venv}/bin/python scripts/plot_eval_compare.py "
        f"{' '.join(series_args)} "
        f"--baseline 0 --title {json.dumps(args.title)} "
        f"--out {args.out_dir / 'compare_all.png'}"
    )
    if plot:
        print(f"\n对比图已保存：{args.out_dir / 'compare_all.png'}", flush=True)
    else:
        print(f"对比图渲染失败（检查 scripts/plot_eval_compare.py 与 matplotlib）", flush=True)


if __name__ == "__main__":
    main()