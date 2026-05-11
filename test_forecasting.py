import torch
import torch.nn as nn
# from utils_dir.Similarity import get_similarity
import numpy as np
import os
import yaml
import models
import argparse
import datetime
import pandas as pd

from models.BiaTCGNet.BiaTCGNet import Model
from data.GenerateDataset import loaddataset

node_number=207
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--task',default='prediction', type=str)
parser.add_argument('--hid_size', type=int)
parser.add_argument('--impute_weight', type=float)
parser.add_argument('--label_weight', type=float)
parser.add_argument("--adj-threshold", type=float, default=0.1)
parser.add_argument('--dataset',default='Metr')
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

parser.add_argument("--config_filename", type=str, default='./models/GTS/para_Metr.yaml')
#####################################################
parser.add_argument("--config", type=str, default='imputation/spin.yaml')
parser.add_argument('--output_attention', type=bool, default=False)
# Splitting/aggregation params
parser.add_argument('--val-len', type=float, default=0.2)
parser.add_argument('--test-len', type=float, default=0.2)
parser.add_argument('--mask_ratio',type=float,default=0.2)
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
parser.add_argument('--seq_len',default=24,type=int)
# parser.add_argument('--history_len',default=24,type=int)
parser.add_argument('--label_len',default=12,type=int)
parser.add_argument('--pred_len',default=24,type=int)
parser.add_argument('--horizon',default=24,type=int)
parser.add_argument('--delay',default=0,type=int)
parser.add_argument('--stride',default=1,type=int)
parser.add_argument('--window_lag',default=1,type=int)
parser.add_argument('--horizon_lag',default=1,type=int)

if __name__ == '__main__':
    args = parser.parse_args()
else:
    args, _ = parser.parse_known_args()
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


def MAPE_np(pred, true, mask_value=0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), (true))))*100


def RMSE_np(pred, true, mask_value=0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

def MAE_np(pred, true, mask_value=0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.abs(pred - true))

def save_metrics_to_csv(metrics_dict, output_dir):
    """保存逐日指标到CSV文件（与HD-TTS格式一致）"""
    os.makedirs(output_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f'metrics_{timestamp}.csv')

    # 创建DataFrame
    df = pd.DataFrame(metrics_dict)

    # 保存CSV
    df.to_csv(csv_path, index=False)
    print(f'\n[OK] 指标已保存到: {csv_path}')

    return csv_path


def test(model, output_metrics_dir=None, return_details=False):
    """测试模型并保存逐日指标（与HD-TTS格式一致）"""
    loss = 0.0
    labels = []
    preds = []
    masks = []

    # 创建输出目录（与HD-TTS格式一致）
    if output_metrics_dir is None:
        exp_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{exp_name}_{args.seed}_dataset-{args.dataset}_seq{args.seq_len}_pred{args.pred_len}_miss{args.mask_ratio}_bs{args.batch_size}_ep200"
        output_metrics_dir = os.path.join('./output_metrics', exp_name)

    os.makedirs(output_metrics_dir, exist_ok=True)

    train_dataloader,val_dataloader, test_dataloader, scaler = loaddataset(args.seq_len,args.pred_len,args.mask_ratio,args.dataset)
    model.eval()
    k=0

    print(f'\n{"="*80}')
    print(f'BiTGraph测试 - 逐日评估（与HD-TTS格式一致）')
    print(f'{"="*80}')
    print(f'数据集: {args.dataset}')
    print(f'缺失率: {args.mask_ratio*100}%')
    print(f'输出目录: {output_metrics_dir}')
    print(f'{"="*80}\n')

    with torch.no_grad():
        for i, (x, y, mask, target_mask) in enumerate(test_dataloader):
            x, y, mask,target_mask = x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()
            x_hat=model(x,mask,k)
            k=k+1
            x_hat=scaler.inverse_transform(x_hat)
            y=scaler.inverse_transform(y)

            preds.append(x_hat.squeeze())
            labels.append(y.squeeze())
            masks.append(target_mask.squeeze())
            losses = torch.sum(torch.abs(x_hat-y)*(target_mask))/torch.sum(target_mask)
            loss+=losses

        labels = torch.cat(labels,dim=0).cpu().numpy()  # (B, H, N)
        preds = torch.cat(preds,dim=0).cpu().numpy()    # (B, H, N)
        y_mask = torch.cat(masks,dim=0).cpu().numpy()   # (B, H, N)

        mask_loss_value = float((loss / len(test_dataloader)).item())

        print(f'整体指标:')
        print(f'  Mask Loss: {mask_loss_value:.4f}')

        # 计算“整体口径（masked）”：在所有可观测位置一次性统计
        def masked_mae_np(pred, true, mask):
            m = mask.reshape(-1) > 0.5
            return float(np.abs(pred.reshape(-1)[m] - true.reshape(-1)[m]).mean())
        def masked_rmse_np(pred, true, mask):
            m = mask.reshape(-1) > 0.5
            err = pred.reshape(-1)[m] - true.reshape(-1)[m]
            return float(np.sqrt(np.mean(err * err)))
        def masked_mape_np(pred, true, mask, eps=1e-8):
            m = mask.reshape(-1) > 0.5
            t = true.reshape(-1)[m]
            p = pred.reshape(-1)[m]
            denom = np.clip(np.abs(t), eps, None)
            return float(np.mean(np.abs((p - t) / denom)) * 100.0)

        overall_mae = masked_mae_np(preds, labels, y_mask)
        overall_rmse = masked_rmse_np(preds, labels, y_mask)
        overall_mape = masked_mape_np(preds, labels, y_mask)
        print(f'  MAE: {overall_mae:.4f}')
        print(f'  RMSE: {overall_rmse:.4f}')
        print(f'  MAPE: {overall_mape:.2f}%\n')

        # 同步保存整体指标到 metrics_overall.csv（与ReCTSI一致）
        overall_path = os.path.join(output_metrics_dir, 'metrics_overall.csv')
        pd.DataFrame([{'mae': overall_mae, 'rmse': overall_rmse, 'mape': overall_mape}]).to_csv(overall_path, index=False)

        # 逐日评估（与HD-TTS一致）
        print(f'逐日指标:')
        print(f'{"Day":<5} {"MAE":<10} {"RMSE":<10} {"MAPE":<10}')
        print(f'{"-"*40}')

        metrics_dict = {'day': [], 'mae': [], 'rmse': [], 'mape': []}

        for day in range(args.pred_len):
            # 提取第day天的预测和真实值
            day_preds = preds[:, day, :]  # (B, N)
            day_labels = labels[:, day, :]  # (B, N)

            # 计算指标
            day_mae = MAE_np(day_preds, day_labels)
            day_rmse = RMSE_np(day_preds, day_labels)
            day_mape = MAPE_np(day_preds, day_labels)

            metrics_dict['day'].append(day + 1)
            metrics_dict['mae'].append(day_mae)
            metrics_dict['rmse'].append(day_rmse)
            metrics_dict['mape'].append(day_mape)

            print(f'{day+1:<5} {day_mae:<10.4f} {day_rmse:<10.4f} {day_mape:<10.2f}')

        # 保存CSV
        csv_path = save_metrics_to_csv(metrics_dict, output_metrics_dir)

        print(f'\n{"="*80}')
        print(f'测试完成！')
        print(f'{"="*80}\n')

    details = {
        'mask_loss': mask_loss_value,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_mape': overall_mape,
        'overall_metrics_path': overall_path,
        'daily_metrics_path': csv_path,
        'output_metrics_dir': output_metrics_dir,
    }

    if return_details:
        return details
    return overall_mae

def run():



    model = Model(True, True, 2, node_number,args.kernel_set,
                  'cuda:0', predefined_A=None,
                  dropout=0.3, subgraph_size=5,
                  node_dim=3,
                  dilation_exponential=1,
                  conv_channels=8, residual_channels=8,
                  skip_channels=16, end_channels=32,
                  seq_length=args.seq_len, in_dim=1, out_len=args.pred_len, out_dim=1,
                  layers=2, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).cuda()

    model.load_state_dict(torch.load('./output_BiaTCGNet_'+args.dataset+'_miss'+str(args.mask_ratio)+'_'+args.task+'/best.pth'))
    loss=test(model)

    print('loss:',loss)

if __name__ == '__main__':
    run()
