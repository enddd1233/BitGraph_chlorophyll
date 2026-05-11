"""
BiTGraph批量训练脚本
与HD-TTS格式完全对齐

训练6个实验：
1. Bohai 20%
2. Bohai 40%
3. Bohai 60%
4. Nanhai 20%
5. Nanhai 40%
6. Nanhai 60%
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import datetime
import argparse
import pandas as pd

def clean_old_logs():
    """删除旧的训练日志"""
    dirs_to_clean = [
        './xiaorongshiyan/log_dir',
        './xiaorongshiyan/output_models',
        './xiaorongshiyan/output_metrics',
        './output_BiaTCGNet_Bohai_miss0.2_prediction',
        './output_BiaTCGNet_Bohai_miss0.4_prediction',
        './output_BiaTCGNet_Nanhai_miss0.2_prediction',
        './output_BiaTCGNet_Nanhai_miss0.4_prediction',
    ]
    
    print(f'\n{"="*80}')
    print(f'清理旧的训练日志...')
    print(f'{"="*80}\n')
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f'[OK] 已删除: {dir_path}')
            except Exception as e:
                print(f'[WARN] 删除失败: {dir_path} - {e}')
        else:
            print(f'[SKIP] 不存在: {dir_path}')
    
    print(f'\n{"="*80}')
    print(f'清理完成！')
    print(f'{"="*80}\n')


def _find_exp_dir(base_dir, seed, dataset, mask_ratio, batch_size, epochs, ablation, seq_len, pred_len):
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    try:
        miss_percent = int(round(float(mask_ratio) * 100))
    except Exception:
        miss_percent = None

    if miss_percent is not None:
        exp_name = f"{dataset}_{miss_percent}_{seed}"
        outer = base_path / str(ablation)
        candidate = outer / exp_name
        if candidate.exists() and candidate.is_dir():
            return candidate

    suffix = f"_{seed}_dataset-{dataset}_seq{seq_len}_pred{pred_len}_miss{mask_ratio}_bs{batch_size}_ep{epochs}_abl-{ablation}"
    candidates = []
    outer = base_path / str(ablation)
    if outer.exists():
        candidates.extend([p for p in outer.iterdir() if p.is_dir() and p.name.endswith(suffix)])
    candidates.extend([p for p in base_path.iterdir() if p.is_dir() and p.name.endswith(suffix)])
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def run_experiment(dataset, mask_ratio, ablation, seed, batch_size=32, epochs=200, patience=40, lr=0.001, seq_len=30, pred_len=15):
    """运行单个实验"""
    print(f'\n{"="*80}')
    print(f'开始训练: {dataset} - 缺失率{mask_ratio*100}% - 消融{ablation} - seed{seed}')
    print(f'{"="*80}\n')
    
    cmd = [
        sys.executable, 'main.py',
        '--dataset', dataset,
        '--seq_len', str(seq_len),
        '--pred_len', str(pred_len),
        '--mask_ratio', str(mask_ratio),
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--patience', str(patience),
        '--lr', str(lr),
        '--seed', str(seed),
        '--ablation', str(ablation)
    ]
    
    print(f'命令: {" ".join(cmd)}\n')
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f'\n[OK] {dataset} {mask_ratio*100}% {ablation} seed{seed} 训练完成！\n')

        metrics_dir = _find_exp_dir(
            base_dir='./xiaorongshiyan/output_metrics',
            seed=seed,
            dataset=dataset,
            mask_ratio=str(mask_ratio),
            batch_size=batch_size,
            epochs=epochs,
            ablation=ablation,
            seq_len=seq_len,
            pred_len=pred_len,
        )
        return True, metrics_dir
    except subprocess.CalledProcessError as e:
        print(f'\n[FAIL] {dataset} {mask_ratio*100}% {ablation} seed{seed} 训练失败！')
        print(f'错误: {e}\n')
        return False, None


def _aggregate_metrics(rows, output_path):
    if not rows:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['Bohai', 'Nanhai'])
    parser.add_argument('--mask_ratios', nargs='+', type=float, default=[0.2, 0.4, 0.6])
    parser.add_argument('--ablations', nargs='+', default=['tcgnet', 'wo_msipt', 'wo_bgcn', 'wo_eq4', 'wo_eq9'])
    parser.add_argument('--include_full', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=[999])
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--pred_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()

    if args.include_full and 'bitgraph' not in args.ablations:
        args.ablations = ['bitgraph'] + list(args.ablations)

    print(f'\n{"#"*80}')
    print(f'# BiTGraph批量训练脚本')
    print(f'# 与HD-TTS格式完全对齐')
    print(f'# 开始时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"#"*80}\n')
    
    # 1. 清理旧日志
    if args.clean:
        clean_old_logs()
    
    # 2. 定义实验列表
    experiments = []
    for dataset in args.datasets:
        for mask_ratio in args.mask_ratios:
            for ablation in args.ablations:
                for seed in args.seeds:
                    experiments.append({
                        'dataset': dataset,
                        'mask_ratio': mask_ratio,
                        'ablation': ablation,
                        'seed': seed,
                        'name': f'{dataset} {int(mask_ratio*100)}% {ablation} seed{seed}',
                    })
    
    # 3. 运行所有实验
    results = []
    for i, exp in enumerate(experiments, 1):
        print(f'\n{"#"*80}')
        print(f'# 实验 {i}/{len(experiments)}: {exp["name"]}')
        print(f'{"#"*80}')
        
        success, metrics_dir = run_experiment(
            dataset=exp['dataset'],
            mask_ratio=exp['mask_ratio'],
            ablation=exp['ablation'],
            seed=exp['seed'],
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
        )
        
        results.append({
            'name': exp['name'],
            'dataset': exp['dataset'],
            'mask_ratio': exp['mask_ratio'],
            'ablation': exp['ablation'],
            'seed': exp['seed'],
            'metrics_dir': str(metrics_dir) if metrics_dir is not None else None,
            'success': success
        })
    
    # 4. 打印总结
    print(f'\n{"#"*80}')
    print(f'# 所有实验完成！')
    print(f'# 完成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"#"*80}\n')
    
    print(f'实验结果总结:')
    print(f'{"-"*80}')
    for i, result in enumerate(results, 1):
        status = '[OK] 成功' if result['success'] else '[FAIL] 失败'
        print(f'{i}. {result["name"]:<20} {status}')
    print(f'{"-"*80}\n')
    
    # 5. 检查输出目录
    print(f'输出目录:')
    print(f'{"-"*80}')
    
    for dir_name in ['log_dir', 'output_models', 'output_metrics']:
        dir_path = Path('./xiaorongshiyan') / dir_name
        if dir_path.exists():
            experiment_dirs = [p for p in dir_path.rglob('*') if p.is_dir() and p.parent.is_dir() and p.parent.name in args.ablations]
            if not experiment_dirs:
                experiment_dirs = [p for p in dir_path.iterdir() if p.is_dir()]
            print(f'[DIR] {dir_name}: {len(experiment_dirs)} 个实验')
            shown = 0
            for subdir in sorted(experiment_dirs):
                print(f'   - {subdir}')
                shown += 1
                if shown >= 10:
                    break
            if len(experiment_dirs) > 10:
                print('   - ...')
        else:
            print(f'[DIR] {dir_name}: 不存在')
    
    print(f'{"-"*80}\n')
    
    # 6. 统计成功率
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    success_rate = success_count / total_count * 100
    
    print(f'总体统计:')
    print(f'{"-"*80}')
    print(f'总实验数: {total_count}')
    print(f'成功数: {success_count}')
    print(f'失败数: {total_count - success_count}')
    print(f'成功率: {success_rate:.1f}%')
    print(f'{"-"*80}\n')

    # 7. 聚合整体指标
    if args.aggregate:
        metric_rows = []
        for r in results:
            if not r['success']:
                continue
            if not r.get('metrics_dir'):
                continue
            overall_path = Path(r['metrics_dir']) / 'metrics_overall.csv'
            if not overall_path.exists():
                continue

            try:
                overall_df = pd.read_csv(overall_path)
                if overall_df.empty:
                    continue
                row = overall_df.iloc[0].to_dict()
                row.update({
                    'dataset': r['dataset'],
                    'mask_ratio': r['mask_ratio'],
                    'ablation': r['ablation'],
                    'seed': r['seed'],
                    'exp_name': Path(r['metrics_dir']).name,
                })
                metric_rows.append(row)
            except Exception:
                continue

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join('./xiaorongshiyan', 'output_metrics', f'summary_{timestamp}.csv')
        saved_path = _aggregate_metrics(metric_rows, summary_path)
        if saved_path is not None:
            print(f'[OK] 聚合结果已保存: {saved_path}')

if __name__ == '__main__':
    main()

