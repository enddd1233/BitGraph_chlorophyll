import configparser
import copy
import json
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.BiaTCGNet.BiaTCGNet import Model
import models
import argparse
import os
import sys
import yaml
from data.GenerateDataset import loaddataset
# from tsl.data.utils import WINDOW
import datetime

TIME_CONTRAST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if TIME_CONTRAST_ROOT not in sys.path:
    sys.path.insert(0, TIME_CONTRAST_ROOT)

from measurement_utils import IterationMeasurer, StopMeasurement, write_iter_times_csv

torch.multiprocessing.set_sharing_strategy('file_system')
node_number=207
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--task', default='prediction',type=str)
parser.add_argument("--adj-threshold", type=float, default=0.1)
parser.add_argument('--dataset',default='Elec')
parser.add_argument('--val_ratio',default=0.2)
parser.add_argument('--test_ratio',default=0.2)
parser.add_argument('--column_wise',default=False)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument("--model-name", type=str, default='spin')
parser.add_argument("--dataset-name", type=str, default='air36'
                                                        '')
parser.add_argument('--fc_dropout', default=0.2, type=float)
parser.add_argument('--head_dropout', default=0, type=float)
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--kernel_set', type=list, default=[2,3,6,7], help='kernel set')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--conv_channels', type=int, default=8)
parser.add_argument('--residual_channels', type=int, default=8)
parser.add_argument('--skip_channels', type=int, default=16)
parser.add_argument('--end_channels', type=int, default=32)
parser.add_argument('--model_dropout', type=float, default=0.3)
parser.add_argument('--gcn_depth', type=int, default=2)
parser.add_argument('--subgraph_size', type=int, default=5)
parser.add_argument('--node_dim', type=int, default=3)
parser.add_argument('--dilation_exp', type=int, default=1)
parser.add_argument('--mask_topk', type=int, default=10)
parser.add_argument('--mask_bias_w', type=float, default=0.003)
parser.add_argument('--tri_bias_w', type=float, default=0.0015)
parser.add_argument('--ablation', type=str, default='bitgraph', choices=['bitgraph', 'wo_adp', 'tcgnet', 'wo_msipt', 'wo_bgcn', 'wo_eq4', 'wo_eq9'])
parser.add_argument('--output_root', type=str, default='./xiaorongshiyan')
parser.add_argument('--flat_output_layout', action='store_true', help='write log_dir/output_models/output_metrics directly under output_root')
parser.add_argument('--export_adp', action='store_true')
parser.add_argument('--export_adp_epochs', type=str, default='1,10,30,60,90')
parser.add_argument('--export_adp_dir', type=str, default='ADP')
parser.add_argument('--measure_only', action='store_true')
parser.add_argument('--warmup_iters', type=int, default=20)
parser.add_argument('--measure_iters', type=int, default=30)
parser.add_argument('--measure_output', type=str, default=None)
##############transformer config############################

parser.add_argument('--enc_in', type=int, default=node_number, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=node_number, help='decoder input size')
parser.add_argument('--c_out', type=int, default=node_number, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--num_nodes', type=int, default=node_number, help='dimension of fcn')
parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')
#######################AGCRN##########################
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--output_dim', default=1, type=int)
parser.add_argument('--embed_dim', default=512, type=int)
parser.add_argument('--rnn_units', default=64, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--cheb_k', default=2, type=int)
parser.add_argument('--default_graph', type=bool, default=True)

#############GTS##################################
parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')

parser.add_argument("--config_filename", type=str, default='')
#####################################################
parser.add_argument("--config", type=str, default='imputation/spin.yaml')
parser.add_argument('--output_attention', type=bool, default=False)
# Splitting/aggregation params
parser.add_argument('--val-len', type=float, default=0.2)
parser.add_argument('--test-len', type=float, default=0.2)
parser.add_argument('--mask_ratio',type=float,default=0.1)
# Training params
parser.add_argument('--lr', type=float, default=0.001)  #0.001
# parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=40)
parser.add_argument('--l2-reg', type=float, default=0.)
# parser.add_argument('--batches-epoch', type=int, default=300)
parser.add_argument('--batch-inference', type=int, default=32)
parser.add_argument('--split-batch-in', type=int, default=1)
parser.add_argument('--grad-clip-val', type=float, default=5.)
parser.add_argument('--loss-fn', type=str, default='l1_loss')
parser.add_argument('--lr-scheduler', type=str, default=None)
parser.add_argument('--seq_len',default=24,type=int) # 96
# parser.add_argument('--history_len',default=24,type=int) #96
parser.add_argument('--label_len',default=12,type=int) #48
parser.add_argument('--pred_len',default=24,type=int)
parser.add_argument('--horizon',default=24,type=int)
parser.add_argument('--delay',default=0,type=int)
parser.add_argument('--stride',default=1,type=int)
parser.add_argument('--window_lag',default=1,type=int)
parser.add_argument('--horizon_lag',default=1,type=int)

# Connectivity params
# parser.add_argument("--adj-threshold", type=float, default=0.1)
args = parser.parse_args()
criteron=nn.L1Loss().cuda()

if(args.dataset=='Metr'):
    node_number=207
    args.num_nodes=207
    args.enc_in=207
    args.dec_in=207
    args.c_out=207
elif(args.dataset=='PEMS'):
    node_number=325
    args.num_nodes=325
    args.enc_in = 325
    args.dec_in = 325
    args.c_out = 325
elif(args.dataset=='ETTh1'):
    node_number=7
    args.num_nodes=7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
elif(args.dataset=='Elec'):
    node_number=321
    args.num_nodes=321
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
elif(args.dataset=='BeijingAir'):
    node_number=36
    args.num_nodes=36
    args.enc_in = 36
    args.dec_in = 36
    args.c_out = 36
elif(args.dataset=='Bohai'):
    node_number=300
    args.num_nodes=300
    args.enc_in = 300
    args.dec_in = 300
    args.c_out = 300
elif(args.dataset=='Nanhai'):
    node_number=265
    args.num_nodes=265
    args.enc_in = 265
    args.dec_in = 265
    args.c_out = 265


def _serialize_for_json(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_for_json(val) for key, val in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _args_snapshot(namespace):
    return {key: _serialize_for_json(value) for key, value in sorted(vars(namespace).items())}


def _write_config_snapshot(config_path, namespace):
    args_dict = _args_snapshot(namespace)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("BiTGraph训练配置\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"数据集: {args_dict.get('dataset')}\n")
        f.write(f"节点数: {args_dict.get('num_nodes')}\n")
        f.write(f"输入序列长度: {args_dict.get('seq_len')}\n")
        f.write(f"预测序列长度: {args_dict.get('pred_len')}\n")
        f.write(f"缺失率: {float(args_dict.get('mask_ratio', 0.0)) * 100:.1f}%\n")
        f.write(f"批次大小: {args_dict.get('batch_size')}\n")
        f.write(f"最大轮数: {args_dict.get('epochs')}\n")
        f.write(f"早停耐心: {args_dict.get('patience')}\n")
        f.write(f"学习率: {args_dict.get('lr')}\n")
        f.write(f"随机种子: {args_dict.get('seed')}\n")
        f.write(f"消融设置: {args_dict.get('ablation')}\n")
        f.write(f"输出根目录: {args_dict.get('output_root')}\n")
        f.write(f"扁平输出布局: {args_dict.get('flat_output_layout')}\n")
        f.write(f"\n{'='*80}\n")
        f.write("完整参数\n")
        f.write(f"{'-'*80}\n")
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")


def _write_best_summary(summary_path, train_result, test_result):
    payload = {
        'best_epoch': int(train_result['best_epoch']),
        'best_val_loss': float(train_result['best_loss']),
        'log_dir': train_result['logdir'],
        'model_dir': train_result['modeldir'],
        'metrics_dir': train_result['metricsdir'],
        'history_csv': train_result.get('history_path'),
        'config_path': train_result.get('config_path'),
        'training_log_path': train_result.get('training_log_path'),
        'training_summary_path': train_result.get('summary_path'),
        'best_epoch_path': train_result.get('best_epoch_path'),
        'test_mask_loss': float(test_result['mask_loss']),
        'test_mae': float(test_result['overall_mae']),
        'test_rmse': float(test_result['overall_rmse']),
        'test_mape': float(test_result['overall_mape']),
        'metrics_overall_csv': test_result.get('overall_metrics_path'),
        'daily_metrics_csv': test_result.get('daily_metrics_path'),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _build_distance_knn_adjacency_from_csv(csv_path, num_nodes, k=10, sym=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)
    if ('lat' not in df.columns) or ('lon' not in df.columns):
        raise ValueError(f"CSV missing required columns 'lat'/'lon'. Available columns: {list(df.columns)}")

    coords = df[['lat', 'lon']].iloc[:num_nodes].to_numpy(dtype=np.float64)
    if coords.shape[0] < num_nodes:
        raise ValueError(f'CSV has {coords.shape[0]} rows, expected at least {num_nodes} nodes')

    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2.0) ** 2) + (np.cos(lat[:, None]) * np.cos(lat[None, :]) * (np.sin(dlon / 2.0) ** 2))
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    dist = 6371.0 * c
    np.fill_diagonal(dist, np.inf)

    k = int(k)
    if k <= 0 or k >= int(num_nodes):
        raise ValueError(f'Invalid k={k} for num_nodes={num_nodes}')

    nbr_idx = np.argpartition(dist, kth=k, axis=1)[:, :k]
    nbr_dist = dist[np.arange(num_nodes)[:, None], nbr_idx]

    sigma = float(np.mean(nbr_dist))
    if (not np.isfinite(sigma)) or (sigma <= 0.0):
        sigma = 1.0
    sigma = sigma + 1e-6
    weights = np.exp(-((nbr_dist / sigma) ** 2))

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    rows = np.repeat(np.arange(num_nodes), k)
    cols = nbr_idx.reshape(-1)
    A[rows, cols] = weights.reshape(-1).astype(np.float32)

    if sym:
        A = np.maximum(A, A.T)

    np.fill_diagonal(A, 0.0)
    return A

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'l2_reg', 0.0))
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)

    # 创建实验名称（与HD-TTS格式一致）
    miss_percent = int(round(args.mask_ratio * 100))
    exp_name = f"{args.dataset}_{miss_percent}_{args.seed}"

    # 创建输出目录
    output_root = str(getattr(args, 'output_root', './xiaorongshiyan'))
    if getattr(args, 'flat_output_layout', False):
        logdir = os.path.join(output_root, 'log_dir')
        modeldir = os.path.join(output_root, 'output_models')
        metricsdir = os.path.join(output_root, 'output_metrics')
    else:
        logdir = os.path.join(output_root, 'log_dir', str(args.ablation), exp_name)
        modeldir = os.path.join(output_root, 'output_models', str(args.ablation), exp_name)
        metricsdir = os.path.join(output_root, 'output_metrics', str(args.ablation), exp_name)

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(modeldir, exist_ok=True)
    os.makedirs(metricsdir, exist_ok=True)

    export_epochs = set()
    if getattr(args, 'export_adp', False):
        for _tok in str(getattr(args, 'export_adp_epochs', '')).split(','):
            _tok = _tok.strip()
            if _tok:
                export_epochs.add(int(_tok))

        export_adp_dir = str(getattr(args, 'export_adp_dir', 'ADP')).strip()
        if not export_adp_dir:
            export_adp_dir = 'ADP'
        if os.path.isabs(export_adp_dir):
            figdir = export_adp_dir
        else:
            figdir = os.path.join(metricsdir, export_adp_dir)
        os.makedirs(figdir, exist_ok=True)

    # 保存配置文件
    config_path = os.path.join(logdir, 'config.txt')
    _write_config_snapshot(config_path, args)

    print(f"[OK] 配置文件已保存: {config_path}")

    train_dataloader, val_dataloader, test_dataloader, scaler=loaddataset(args.seq_len,args.pred_len,args.mask_ratio,args.dataset)

    best_loss=9999999.99
    best_epoch = 0
    patience_counter = 0
    k=0
    history_rows = []
    best_model = copy.deepcopy(model.state_dict())

    adp_order = None
    adp_prev_ordered = None
    adp_prev_epoch = None

    # 训练日志文件
    training_log_path = os.path.join(logdir, 'training_log.txt')
    measure_enabled = bool(args.measure_only and args.measure_output)
    iter_times_path = None
    if args.measure_output:
        iter_times_path = os.path.join(os.path.dirname(os.path.abspath(args.measure_output)), 'iter_times.csv')
    measurer = IterationMeasurer(
        enabled=measure_enabled,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        stage='train',
        iter_type='optimizer_step',
        rate_scope='per_rate',
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
    )

    print(f'\n{"="*80}')
    print(f'BiTGraph训练开始')
    print(f'{"="*80}')
    print(f'数据集: {args.dataset}')
    print(f'缺失率: {args.mask_ratio*100}%')
    print(f'最大轮数: {args.epochs}')
    print(f'早停耐心: {args.patience}')
    print(f'日志目录: {logdir}')
    print(f'模型目录: {modeldir}')
    print(f'指标目录: {metricsdir}')
    print(f'{"="*80}\n')

    # 写入训练日志头部
    with open(training_log_path, 'w', encoding='utf-8') as f:
        f.write(f"BiTGraph训练日志\n")
        f.write(f"{'='*80}\n")
        f.write(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

    measurement_stopped = False
    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_train_loss = 0.0
            epoch_train_steps = 0
            for i, (x, y, mask, target_mask) in enumerate(train_dataloader):

                if measure_enabled:
                    measurer.start_iter()
                x, y, mask,target_mask =x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()
                x=x*mask
                y=y*target_mask
                x_hat=model(x,mask,k)
                batch_loss = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
                optimizer.zero_grad()  # optimizer.zero_grad()
                batch_loss.backward()
                # 梯度裁剪，稳定训练
                try:
                    from torch.nn.utils import clip_grad_norm_
                    clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_val)
                except Exception:
                    pass
                optimizer.step()
                epoch_train_loss += float(batch_loss.item())
                epoch_train_steps += 1
                if measure_enabled:
                    measurer.end_iter(epoch=epoch + 1, iter_in_epoch=i + 1)

            if args.measure_only:
                continue

            train_loss_value = epoch_train_loss / max(epoch_train_steps, 1)
            loss=evaluate(model, val_dataloader,scaler)
            val_loss_value = float(loss.item())

            if getattr(args, 'export_adp', False) and (epoch + 1) in export_epochs:
                with torch.no_grad():
                    _m = model.module if hasattr(model, 'module') else model
                    if getattr(_m, 'buildA_true', True):
                        adp = _m.gc(_m.idx)
                    else:
                        adp = _m.predefined_A
                    adp_np = adp.detach().float().cpu().numpy()

                npy_path = os.path.join(figdir, f'adp_epoch{epoch+1:03d}.npy')
                png_path = os.path.join(figdir, f'adp_epoch{epoch+1:03d}.png')
                np.save(npy_path, adp_np)

                fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                im = ax.imshow(adp_np, cmap='viridis', vmin=0.0, vmax=1.0, interpolation='nearest')
                ax.tick_params(axis='both', labelsize=6)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=6)
                fig.tight_layout()
                fig.savefig(png_path, bbox_inches='tight', pad_inches=0.05)
                plt.close(fig)

                if adp_order is None:
                    strength = np.sum(adp_np, axis=0) + np.sum(adp_np, axis=1)
                    adp_order = np.argsort(-strength)
                    try:
                        np.save(os.path.join(figdir, 'adp_node_order.npy'), adp_order)
                    except Exception:
                        pass

                adp_np_ordered = adp_np[np.ix_(adp_order, adp_order)]
                vals = adp_np_ordered.reshape(-1)
                vals = vals[np.isfinite(vals)]
                vals_nz = vals[vals > 0]
                if vals_nz.size > 0:
                    vmax_contrast = float(np.quantile(vals_nz, 0.99))
                elif vals.size > 0:
                    vmax_contrast = float(np.max(vals))
                else:
                    vmax_contrast = 1.0
                if (not np.isfinite(vmax_contrast)) or (vmax_contrast <= 0.0):
                    vmax_contrast = 1.0

                contrast_path = os.path.join(figdir, f'adp_epoch{epoch+1:03d}_contrast.png')
                fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                im = ax.imshow(adp_np_ordered, cmap='viridis', vmin=0.0, vmax=vmax_contrast, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=6)
                fig.tight_layout()
                fig.savefig(contrast_path, bbox_inches='tight', pad_inches=0.05)
                plt.close(fig)

                if adp_prev_ordered is not None:
                    delta = adp_np_ordered - adp_prev_ordered
                    abs_delta = np.abs(delta.reshape(-1))
                    abs_delta = abs_delta[np.isfinite(abs_delta)]
                    abs_delta_nz = abs_delta[abs_delta > 0]
                    if abs_delta_nz.size > 0:
                        dlim = float(np.quantile(abs_delta_nz, 0.99))
                    elif abs_delta.size > 0:
                        dlim = float(np.max(abs_delta))
                    else:
                        dlim = 0.0
                    if np.isfinite(dlim) and dlim > 0.0:
                        delta_path = os.path.join(figdir, f'adp_delta_epoch{int(adp_prev_epoch):03d}_to_{epoch+1:03d}.png')
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                        im = ax.imshow(delta, cmap='seismic', vmin=-dlim, vmax=dlim, interpolation='nearest')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=6)
                        fig.tight_layout()
                        fig.savefig(delta_path, bbox_inches='tight', pad_inches=0.05)
                        plt.close(fig)

                adp_prev_ordered = adp_np_ordered
                adp_prev_epoch = epoch + 1

            # 写入训练日志
            is_best = val_loss_value < best_loss
            history_rows.append({
                'epoch': epoch,
                'train_loss': train_loss_value,
                'val_loss': val_loss_value,
                'is_best': int(is_best),
            })

            log_msg = f'Epoch {epoch:3d}, Train Loss: {train_loss_value:.6f}, Val Loss: {val_loss_value:.6f}'
            print(log_msg)
            with open(training_log_path, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')

            if is_best:
                best_loss = val_loss_value
                best_epoch = epoch
                patience_counter = 0
                best_model = copy.deepcopy(model.state_dict())

                # 保存到新的目录结构
                model_path = os.path.join(modeldir, 'best.pth')
                torch.save(best_model, model_path)

                msg = f'  [OK] 新的最佳模型！Epoch {epoch}, Loss: {val_loss_value:.4f}'
                print(msg)
                with open(training_log_path, 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    msg = f'\n{"="*80}\n早停触发！连续{args.patience}个epoch无改善\n最佳Epoch: {best_epoch}, 最佳Loss: {best_loss:.4f}\n{"="*80}\n'
                    print(msg)
                    with open(training_log_path, 'a', encoding='utf-8') as f:
                        f.write(msg)
                    break
    except StopMeasurement:
        measurement_stopped = True
    finally:
        if measure_enabled and iter_times_path:
            write_iter_times_csv(iter_times_path, measurer.records)
        if measure_enabled and args.measure_output:
            measurer.write_measure_json(
                args.measure_output,
                extra={
                    'model': 'BiTGraph',
                    'dataset': args.dataset,
                    'missing_rate': args.mask_ratio,
                    'seed': args.seed,
                    'batch_size': args.batch_size,
                    'seq_len': args.seq_len,
                    'pred_len': args.pred_len,
                    'output_root': output_root,
                    'measurement_stopped': measurement_stopped,
                },
            )

    if args.measure_only:
        return {
            'logdir': logdir,
            'modeldir': modeldir,
            'metricsdir': metricsdir,
            'exp_name': exp_name,
            'output_root': output_root,
            'config_path': config_path,
            'training_log_path': training_log_path,
            'history_path': None,
            'best_epoch_path': None,
            'summary_path': None,
            'best_epoch': None,
            'best_loss': None,
        }

    history_path = os.path.join(logdir, 'history.csv')
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    # 保存最佳epoch信息
    best_epoch_path = os.path.join(logdir, 'best_epoch.txt')
    with open(best_epoch_path, 'w', encoding='utf-8') as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Loss: {best_loss:.6f}\n")

    # 保存训练总结
    summary_path = os.path.join(logdir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"BiTGraph训练总结\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"节点数: {args.num_nodes}\n")
        f.write(f"缺失率: {args.mask_ratio*100}%\n")
        f.write(f"总轮数: {epoch + 1}\n")
        f.write(f"最佳轮数: {best_epoch}\n")
        f.write(f"最佳Loss: {best_loss:.6f}\n")
        f.write(f"\n完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")

    print(f'\n{"="*80}')
    print(f'训练完成！')
    print(f'最佳Epoch: {best_epoch}, 最佳Loss: {best_loss:.4f}')
    print(f'[OK] 训练日志: {training_log_path}')
    print(f'[OK] 历史指标: {history_path}')
    print(f'[OK] 最佳epoch: {best_epoch_path}')
    print(f'[OK] 训练总结: {summary_path}')
    print(f'[OK] 模型文件: {os.path.join(modeldir, "best.pth")}')
    print(f'{"="*80}\n')

    # 返回目录信息，供测试使用
    return {
        'logdir': logdir,
        'modeldir': modeldir,
        'metricsdir': metricsdir,
        'exp_name': exp_name,
        'output_root': output_root,
        'config_path': config_path,
        'training_log_path': training_log_path,
        'history_path': history_path,
        'best_epoch_path': best_epoch_path,
        'summary_path': summary_path,
        'best_epoch': best_epoch,
        'best_loss': float(best_loss),
    }


def evaluate(model, val_iter,scaler):
    model.eval()
    loss=0.0
    k=0
    with torch.no_grad():
        for i, (x,y,mask,target_mask) in enumerate(val_iter):
            x, y, mask,target_mask = x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()

            x_hat=model(x,mask,k)

            x_hat = scaler.inverse_transform(x_hat)
            y = scaler.inverse_transform(y)

            losses = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
            loss+=losses


    return loss/len(val_iter)



def run():

    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.ablation == 'wo_bgcn':
        torch.backends.cudnn.enabled = False



    gcn_true = True
    buildA_true = True
    mask_bias_w = args.mask_bias_w
    tri_bias_w = args.tri_bias_w
    temporal_partial_conv = True
    temporal_mask_update = True
    spatial_mask_update = True

    if args.ablation == 'wo_eq4':
        temporal_partial_conv = False
    elif args.ablation == 'wo_adp':
        gcn_true = True
        buildA_true = False
    elif args.ablation == 'wo_eq9':
        mask_bias_w = 0.0
        tri_bias_w = 0.0
    elif args.ablation == 'wo_msipt':
        temporal_partial_conv = False
        temporal_mask_update = False
    elif args.ablation == 'wo_bgcn':
        gcn_true = True
        buildA_true = True
        mask_bias_w = 0.0
        tri_bias_w = 0.0
        spatial_mask_update = False
    elif args.ablation == 'tcgnet':
        temporal_partial_conv = False
        temporal_mask_update = False
        mask_bias_w = 0.0
        tri_bias_w = 0.0
        spatial_mask_update = False

    predefined_A = None
    if args.ablation == 'wo_adp':
        project_root = os.path.dirname(os.path.abspath(__file__))
        if args.dataset == 'Bohai':
            csv_path = os.path.join(project_root, 'Mydata', 'bohai_300.csv')
        elif args.dataset == 'Nanhai':
            csv_path = os.path.join(project_root, 'Mydata', 'nanhai_265.csv')
        else:
            raise ValueError(f'wo_adp requires a dataset with lat/lon CSV (supported: Bohai, Nanhai), got: {args.dataset}')

        A_np = _build_distance_knn_adjacency_from_csv(csv_path, num_nodes=node_number, k=10, sym=True)
        predefined_A = torch.from_numpy(A_np).float().cuda()

    model=Model(gcn_true, buildA_true, args.gcn_depth, node_number,args.kernel_set,
              'cuda:0', predefined_A=predefined_A,
              dropout=args.model_dropout, subgraph_size=args.subgraph_size,
              node_dim=args.node_dim,
              dilation_exponential=args.dilation_exp,
              conv_channels=args.conv_channels, residual_channels=args.residual_channels,
              skip_channels=args.skip_channels, end_channels= args.end_channels,
              seq_length=args.seq_len, in_dim=1,out_len=args.pred_len, out_dim=1,
              layers=args.layers, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
              mask_topk=args.mask_topk, mask_bias_w=mask_bias_w, tri_bias_w=tri_bias_w,
              temporal_partial_conv=temporal_partial_conv, temporal_mask_update=temporal_mask_update, spatial_mask_update=spatial_mask_update)
    if torch.cuda.is_available():
        model = model.cuda()

    train_result = train(model)
    logdir = train_result['logdir']
    modeldir = train_result['modeldir']
    metricsdir = train_result['metricsdir']

    if args.measure_only:
        print('[OK] Measurement-only mode finished.')
        print(f'[DIR] 日志目录: {logdir}')
        print(f'[DIR] 模型目录: {modeldir}')
        print(f'[DIR] 指标目录: {metricsdir}')
        return

    # 训练完成后自动运行测试
    print(f'\n{"="*80}')
    print(f'开始测试...')
    print(f'{"="*80}\n')

    # 加载最佳模型
    model_path = os.path.join(modeldir, 'best.pth')
    model.load_state_dict(torch.load(model_path))

    # 运行测试（会自动保存CSV）
    from test_forecasting import test as test_model
    test_result = test_model(model, output_metrics_dir=metricsdir, return_details=True)

    if getattr(args, 'flat_output_layout', False):
        best_summary_path = os.path.join(train_result['output_root'], 'best_summary.json')
        _write_best_summary(best_summary_path, train_result, test_result)
        print(f'[OK] 最佳结果摘要: {best_summary_path}')

    print(f'\n{"="*80}')
    print(f'实验完成！')
    print(f'{"="*80}')
    print(f'[DIR] 日志目录: {logdir}')
    print(f'[DIR] 模型目录: {modeldir}')
    print(f'[DIR] 指标目录: {metricsdir}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    run()
