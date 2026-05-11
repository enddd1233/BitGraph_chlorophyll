"""
批量评估脚本 - 使用Masked版本的评估指标
评估渤海和南海数据集，缺失率20%/40%/60%的所有模型
结果保存到output_newok目录
"""
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import datetime
import argparse

from models.BiaTCGNet.BiaTCGNet import Model
from data.GenerateDataset import loaddataset


def masked_mae_np(pred, true, mask):
    """Masked MAE - 只在有效观测位置计算"""
    m = mask.reshape(-1) > 0.5
    return float(np.abs(pred.reshape(-1)[m] - true.reshape(-1)[m]).mean())


def masked_rmse_np(pred, true, mask):
    """Masked RMSE - 只在有效观测位置计算"""
    m = mask.reshape(-1) > 0.5
    err = pred.reshape(-1)[m] - true.reshape(-1)[m]
    return float(np.sqrt(np.mean(err * err)))


def masked_mape_np(pred, true, mask, eps=1e-8):
    """Masked MAPE - 只在有效观测位置计算"""
    m = mask.reshape(-1) > 0.5
    t = true.reshape(-1)[m]
    p = pred.reshape(-1)[m]
    denom = np.clip(np.abs(t), eps, None)
    return float(np.mean(np.abs((p - t) / denom)) * 100.0)


def evaluate_model(model_path, dataset, mask_ratio, seq_len, pred_len, output_dir):
    """评估单个模型"""
    
    # 设置节点数
    if dataset == 'Bohai':
        node_number = 300
    elif dataset == 'Nanhai':
        node_number = 265
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f'\n{"="*80}')
    print(f'评估模型: {dataset} - 缺失率{int(mask_ratio*100)}%')
    print(f'{"="*80}')
    print(f'模型路径: {model_path}')
    print(f'输出目录: {output_dir}')
    print(f'{"="*80}\n')
    
    # 创建模型
    model = Model(
        True, True, 2, node_number, [2, 3, 6, 7],
        'cuda:0', predefined_A=None,
        dropout=0.3, subgraph_size=5,
        node_dim=3,
        dilation_exponential=1,
        conv_channels=8, residual_channels=8,
        skip_channels=16, end_channels=32,
        seq_length=seq_len, in_dim=1, out_len=pred_len, out_dim=1,
        layers=2, propalpha=0.05, tanhalpha=3, layer_norm_affline=True
    ).cuda()
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 加载数据
    train_dataloader, val_dataloader, test_dataloader, scaler = loaddataset(
        seq_len, pred_len, mask_ratio, dataset
    )
    
    # 测试
    loss = 0.0
    labels = []
    preds = []
    masks = []
    k = 0
    
    with torch.no_grad():
        for i, (x, y, mask, target_mask) in enumerate(test_dataloader):
            x, y, mask, target_mask = x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()
            x_hat = model(x, mask, k)
            k = k + 1
            x_hat = scaler.inverse_transform(x_hat)
            y = scaler.inverse_transform(y)
            
            preds.append(x_hat.squeeze())
            labels.append(y.squeeze())
            masks.append(target_mask.squeeze())
            losses = torch.sum(torch.abs(x_hat - y) * target_mask) / torch.sum(target_mask)
            loss += losses
    
    labels = torch.cat(labels, dim=0).cpu().numpy()  # (B, H, N)
    preds = torch.cat(preds, dim=0).cpu().numpy()    # (B, H, N)
    y_mask = torch.cat(masks, dim=0).cpu().numpy()   # (B, H, N)
    
    print(f'整体指标 (Masked):')
    print(f'  Mask Loss: {loss/len(test_dataloader):.4f}')
    
    # 计算整体指标 - 使用Masked版本
    overall_mae = masked_mae_np(preds, labels, y_mask)
    overall_rmse = masked_rmse_np(preds, labels, y_mask)
    overall_mape = masked_mape_np(preds, labels, y_mask)
    
    print(f'  MAE (masked): {overall_mae:.4f}')
    print(f'  RMSE (masked): {overall_rmse:.4f}')
    print(f'  MAPE (masked): {overall_mape:.2f}%\n')
    
    # 保存整体指标
    os.makedirs(output_dir, exist_ok=True)
    overall_path = os.path.join(output_dir, 'metrics_overall_masked.csv')
    pd.DataFrame([{
        'dataset': dataset,
        'mask_ratio': mask_ratio,
        'mae': overall_mae,
        'rmse': overall_rmse,
        'mape': overall_mape
    }]).to_csv(overall_path, index=False)
    print(f'[OK] 整体指标已保存: {overall_path}')
    
    # 逐日评估 - 使用Masked版本
    print(f'\n逐日指标 (Masked):')
    print(f'{"Day":<5} {"MAE":<10} {"RMSE":<10} {"MAPE":<10}')
    print(f'{"-"*40}')
    
    metrics_dict = {'day': [], 'mae': [], 'rmse': [], 'mape': []}
    
    for day in range(pred_len):
        # 提取第day天的预测、真实值和mask
        day_preds = preds[:, day, :]    # (B, N)
        day_labels = labels[:, day, :]  # (B, N)
        day_masks = y_mask[:, day, :]   # (B, N)
        
        # 使用Masked版本计算指标
        day_mae = masked_mae_np(day_preds, day_labels, day_masks)
        day_rmse = masked_rmse_np(day_preds, day_labels, day_masks)
        day_mape = masked_mape_np(day_preds, day_labels, day_masks)
        
        metrics_dict['day'].append(day + 1)
        metrics_dict['mae'].append(day_mae)
        metrics_dict['rmse'].append(day_rmse)
        metrics_dict['mape'].append(day_mape)
        
        print(f'{day+1:<5} {day_mae:<10.4f} {day_rmse:<10.4f} {day_mape:<10.2f}')
    
    # 保存逐日指标
    daily_path = os.path.join(output_dir, 'metrics_daily_masked.csv')
    pd.DataFrame(metrics_dict).to_csv(daily_path, index=False)
    print(f'\n[OK] 逐日指标已保存: {daily_path}')
    
    print(f'\n{"="*80}')
    print(f'评估完成！')
    print(f'{"="*80}\n')
    
    return overall_mae, overall_rmse, overall_mape


def main():
    """批量评估所有模型"""
    
    # 配置
    base_model_dir = './output_models'
    base_output_dir = './output_ok1'
    seq_len = 30
    pred_len = 15
    
    # 模型配置
    model_configs = [
        # Bohai
        ('20251101_233000_927246995_dataset-Bohai_seq30_pred15_miss0.2_bs32_ep200', 'Bohai', 0.2),
        ('20251101_233605_59562251_dataset-Bohai_seq30_pred15_miss0.4_bs32_ep200', 'Bohai', 0.4),
        ('20251101_234011_494609771_dataset-Bohai_seq30_pred15_miss0.6_bs32_ep200', 'Bohai', 0.6),
        # Nanhai
        ('20251101_234355_58482903_dataset-Nanhai_seq30_pred15_miss0.2_bs32_ep200', 'Nanhai', 0.2),
        ('20251101_234833_674059135_dataset-Nanhai_seq30_pred15_miss0.4_bs32_ep200', 'Nanhai', 0.4),
        ('20251101_235427_459457166_dataset-Nanhai_seq30_pred15_miss0.6_bs32_ep200', 'Nanhai', 0.6),
    ]
    
    # 汇总结果
    summary_results = []
    
    print('\n' + '='*80)
    print('批量评估开始 - 使用Masked评估指标')
    print(f'总模型数: {len(model_configs)}')
    print('='*80)
    
    for model_dir, dataset, mask_ratio in model_configs:
        try:
            # 模型路径
            model_path = os.path.join(base_model_dir, model_dir, 'best.pth')
            
            if not os.path.exists(model_path):
                print(f'[WARNING] 模型不存在: {model_path}')
                continue
            
            # 输出目录
            output_dir = os.path.join(
                base_output_dir,
                f'{dataset}_miss{int(mask_ratio*100)}'
            )
            
            # 评估模型
            mae, rmse, mape = evaluate_model(
                model_path, dataset, mask_ratio, seq_len, pred_len, output_dir
            )
            
            # 记录结果
            summary_results.append({
                'dataset': dataset,
                'mask_ratio': f'{int(mask_ratio*100)}%',
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            
        except Exception as e:
            print(f'[ERROR] 评估失败: {model_dir}')
            print(f'错误信息: {str(e)}')
            continue
    
    # 保存汇总结果
    if summary_results:
        summary_path = os.path.join(base_output_dir, 'summary_all_masked.csv')
        pd.DataFrame(summary_results).to_csv(summary_path, index=False)
        
        print('\n' + '='*80)
        print('所有评估完成！')
        print('='*80)
        print('\n汇总结果 (Masked评估):')
        print(pd.DataFrame(summary_results).to_string(index=False))
        print(f'\n[OK] 汇总结果已保存: {summary_path}')
        print('='*80 + '\n')


if __name__ == '__main__':
    main()
