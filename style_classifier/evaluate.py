from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manifest predictions and enforce a target accuracy.")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/style_classifier/manifest_eval/predictions.json"))
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/style_classifier/final_evaluation.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("outputs/style_classifier/final_report.md"),
    )
    args = parser.parse_args()
    report = json.loads(args.predictions.read_text(encoding="utf-8"))
    by_style: dict[str, list[bool]] = defaultdict(list)
    for row in report["predictions"]:
        if row["resolution"] != "missing" and not row["is_duplicate"]:
            by_style[row["expected_style"]].append(bool(row["correct"]))
    summary = {
        "target_accuracy": args.target,
        "unique_accuracy": report["unique_accuracy"],
        "target_met": report["unique_accuracy"] >= args.target,
        "per_style": {
            style: {"correct": sum(values), "count": len(values), "accuracy": sum(values) / len(values)}
            for style, values in sorted(by_style.items())
        },
        "macro_recall": report.get("macro_recall"),
        "missing_rows": report["missing_rows"],
        "repair_candidate_rows": report.get("repair_candidate_rows", 0),
        "detector_miss_rows": report.get("detector_miss_rows", 0),
        "detector_coverage": report.get("detector_coverage"),
        "duplicate_rows": report["duplicate_rows"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    per_style_lines = [
        f"| {style} | {value['correct']} | {value['count']} | {value['accuracy']:.2%} |"
        for style, value in summary["per_style"].items()
    ]
    markdown = "\n".join(
        [
            "# 仪表盘样式识别评测报告",
            "",
            "> 任务标签是数据目录中的 Mxx 编号；该结果是固定清单上的封闭集评测，不等同于新来源仪表的泛化准确率。",
            "",
            f"- 唯一图片准确率：**{summary['unique_accuracy']:.2%}**（目标 {summary['target_accuracy']:.0%}，{'达标' if summary['target_met'] else '未达标'}）",
            f"- 四类宏平均召回：**{summary['macro_recall']:.2%}**",
            f"- 冻结 YOLO 检测覆盖率：**{summary['detector_coverage']:.2%}**",
            f"- YOLO 缺检：{summary['detector_miss_rows']} 行（缺检按错误计）",
            f"- Markdown 路径修复候选：{summary['repair_candidate_rows']} 行；重复：{summary['duplicate_rows']} 行",
            "",
            "| 样式 | 正确 | 唯一图 | 召回率 |",
            "| --- | ---: | ---: | ---: |",
            *per_style_lines,
            "",
            "详细逐图证据见 `manifest_eval/predictions.csv` 和 `manifest_eval/predictions.json`。",
            "YOLO 仅加载用户提供的冻结 `best.pt` 做裁剪，本流程没有训练或微调 YOLO。",
            "",
        ]
    )
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["target_met"] else 2)


if __name__ == "__main__":
    main()
