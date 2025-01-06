import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import torch
from torch import distributed
import warnings
import argparse
from ac_ap_dataloader_dali import dali_dataloader
from all_utils import (is_dist_avail_and_initialized, setup_seed, setup_for_distributed, 
                       load_finetune_checkpoint, MetricLogger)

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy
from functools import partial
from tqdm import tqdm
from timm.utils import accuracy
from torch import inf
import math
from torch.nn.utils import clip_grad_norm_


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim
        
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]
        
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias
        
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)
        
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class AttentiveBlock(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()
        
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)
        
        if drop_path > 0.:
            print(f"Use DropPath in projector: {drop_path}")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        
        return x


class AttentionPoolingBlock(AttentiveBlock):
    
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class CustomModel(nn.Module):
    def __init__(self, attentive_probe_model, 
                 attentive_dim, num_classes,
                 init_scale=0.001):
        super(CustomModel, self).__init__()
        
        self.attentive_probe_model = attentive_probe_model

        self.fc_norm = nn.LayerNorm(attentive_dim)
        self.head = nn.Linear(attentive_dim, num_classes)
        
        self.apply(self._init_weights)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features):
        x = self.attentive_probe_model(features)
        x = self.fc_norm(x)
        x = self.head(x)
        return x


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([
        final_value + 0.5 * (base_value - final_value) *
        (1 + math.cos(math.pi * i / (len(iters)))) for i in iters
    ])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train_AdamW(args,
                lr, ap_model, cur_device,
                forward_base_model, data_loader_train, data_loader_val):

    cur_ap_model = ap_model.to(cur_device)
    cur_ap_model = torch.nn.parallel.DistributedDataParallel(cur_ap_model, 
                                                          device_ids=[args.local_rank])
    cur_ap_model.train()


    optimizer = torch.optim.AdamW(
        cur_ap_model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=[0.9, 0.999],
        weight_decay=args.default_weight_decay,
    )
    lr_schedule_values = cosine_scheduler(
        lr,
        args.default_min_lr,
        args.default_epoch,
        args.num_train_steps_per_epoch,
        warmup_epochs=args.default_warmup_epochs,
        start_warmup_value=args.default_start_warmup_value)
    wd_schedule_values = cosine_scheduler(args.default_weight_decay,
                                        args.default_weight_decay,
                                        args.default_epoch,
                                        args.num_train_steps_per_epoch)

    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(cur_device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(cur_device)

    # start
    for epoch in range(args.default_epoch):
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)
        start_steps = int(epoch * args.num_train_steps_per_epoch)
        
        cur_ap_model.train()
        optimizer.zero_grad()
        
        for data_iter_step, (videos, labels) in enumerate(
                            metric_logger.log_every(data_loader_train, args.print_freq, header,
                                                    args.world_size, args.batch_size)):

            global_step = start_steps + data_iter_step
            # Update LR & WD for the first acc
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[global_step]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[global_step]


            videos = videos.to(cur_device, non_blocking=True)
            labels = labels.to(cur_device, non_blocking=True)
            labels = labels.view(-1)
            # base model export feature
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = forward_base_model(videos)
                    outputs = F.normalize(outputs, dim=-1)
            # attentive_probing
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = cur_ap_model(outputs)
                loss = criterion(pred, labels)
                
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                return 0, 0

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(cur_ap_model.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)
            metric_logger.update(grad_norm=grad_norm)
            
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        # epoch eval
        test_stats = validation_one_epoch(args, cur_ap_model, device, forward_base_model, data_loader_val)

        if args.global_rank == 0:
            head_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format("dataset", "seed", 
                                                                                    "default_epoch", "default_warmup_epochs",
                                                                                    "default_weight_decay", "default_min_lr",
                                                                                    "lr", "cur_epoch", "acc_top1", "acc_top5")
            cur_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format(args.data_set, args.seed, 
                                                                                args.default_epoch, args.default_warmup_epochs, 
                                                                                args.default_weight_decay, args.default_min_lr, 
                                                                                lr, epoch, test_stats["acc1"], test_stats["acc5"])
            with open(args.report_txt_path, "a+") as writer:
                writer.write(head_info)
                writer.write(cur_info)

        data_loader_train.reset()
        data_loader_val.reset()
        
    return test_stats["acc1"], test_stats["acc5"]


@torch.no_grad()
def validation_one_epoch(args, 
                         model, cur_device,
                         forward_base_model, data_loader_val):
    
    metric_logger_val = MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()
    for (videos, target) in metric_logger_val.log_every(data_loader_val, args.print_freq, header,
                                                        args.world_size, args.batch_size):
        videos = videos.to(cur_device, non_blocking = True)
        target = target.to(cur_device, non_blocking = True)
        target = target.view(-1)

        # base model export feature
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = forward_base_model(videos)
                outputs = F.normalize(outputs, dim=-1)

        # attentive_probing
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = model(outputs)

        acc1, acc5 = accuracy(pred, target, topk=(1, 5))
        batch_size = videos.shape[0]
        metric_logger_val.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger_val.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger_val.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'.format(top1=metric_logger_val.acc1, 
                                                                             top5=metric_logger_val.acc5))
    return {k: meter.global_avg for k, meter in metric_logger_val.meters.items()}


def find_peak(args, lr, ap_model, device,
            forward_base_model, data_loader_train, data_loader_val):

    acc_top1, acc_top5 = train_AdamW(args, lr, ap_model, device,
                                    forward_base_model, data_loader_train, data_loader_val)
    return acc_top1, acc_top5


def get_args():
    parser = argparse.ArgumentParser('Extract features using the videomae model', add_help=False)
    parser.add_argument('--data_root_path', default="/mnt2/video_pretrain_dataset")
    parser.add_argument('--data_csv_path', default="/mnt2/video_pretrain_dataset/annotation")
    parser.add_argument('--save_report', default="fewshot_video_report/ActionRecognition")

    parser.add_argument('--data_set', default="K400")
    parser.add_argument('--num_shots', default=10, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_step", default=8, type=int)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--model_name', default='umt')
    parser.add_argument('--model', default='vit_base_patch16_224')
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--finetune', default='checkpoint/umt/umt-B-16____K710_65W_epoch200.pth')
    parser.add_argument("--num_frames", default=8, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--tubelet_size", default=1, type=int)
    parser.add_argument("--embedding_size", default=768, type=int)

    # default
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float)
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float)
    parser.add_argument('--dali_num_threads', default=8, type=int)
    parser.add_argument('--dali_py_num_workers', default=14, type=int)
    parser.add_argument('--short_side_size', default=256, type=int)
    parser.add_argument('--use_rgb', default=False)
    parser.add_argument('--smoothing', default=0.1, type=float)

    # default
    parser.add_argument('--default_warmup_epochs', default=5, type=int)
    parser.add_argument('--default_epoch', default=20, type=int)
    parser.add_argument('--default_attentive_head', default=16, type=int)
    parser.add_argument('--default_attentive_out_dim', default=768, type=int)
    parser.add_argument('--default_weight_decay', default=1e-4, type=float)
    parser.add_argument('--default_min_lr', default=1e-7, type=float)
    parser.add_argument('--default_lr_list', default=[1e-3, 3e-4, 1e-4], type=float)
    parser.add_argument('--default_start_warmup_value', default=0.0, type=float)
    parser.add_argument('--clip_grad', default=5.0, type=float)
    parser.add_argument('--print_freq', default=20, type=int)
    return parser.parse_args()


def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    args = get_args()

    # check num_classes
    nb_classes_map = {
        'K400': 400,
        'K600': 600,
        'K700': 700,
        'K710': 710,
        'SSV2': 174,
        'UCF101': 101,
        'HMDB51': 51,
        'Diving48': 48,
        'MIT': 339
    }
    args.num_classes = nb_classes_map[args.data_set]

    try:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl")
    except KeyError:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        distributed.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=args.rank,
            world_size=args.world_size,
        )

    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.local_rank)

    args.global_rank = args.rank
    setup_for_distributed(args.global_rank == 0)
    setup_seed(seed=args.seed, cuda_deterministic=False)

    if args.global_rank == 0:
        mkdir_os(args.save_report)
        mkdir_os(os.path.join(args.save_report, args.data_set))
        args.report_txt_path = os.path.join(args.save_report, args.data_set, "report_attentive_probe_{}.txt".format(args.model_name))
        head_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format("model_name", "model", "num_frames", 
                                                                            "input_size", "tubelet_size", "finetune")
        cur_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<30}\n".format(args.model_name, args.model, args.num_frames, 
                                                                            args.input_size, args.tubelet_size, args.finetune)
        with open(args.report_txt_path, "a+") as writer:
            writer.write(head_info)
            writer.write(cur_info)


    print("create data loader start")
    data_loader_train = dali_dataloader(args.data_root_path,
                                        args.data_csv_path,
                                        args.data_set,
                                        dali_num_threads=args.dali_num_threads,
                                        dali_py_num_workers=args.dali_py_num_workers,
                                        batch_size=args.batch_size,
                                        input_size=args.input_size,
                                        short_side_size=args.short_side_size,
                                        sequence_length=args.num_frames,
                                        use_rgb=args.use_rgb,
                                        mean=args.mean,
                                        std=args.std,
                                        mode="train",
                                        seed=args.seed)
    args.total_batch_size = args.world_size * args.batch_size
    args.num_train_steps_per_epoch = len(data_loader_train)
    
    data_loader_val = dali_dataloader(args.data_root_path,
                                        args.data_csv_path,
                                        args.data_set,
                                        dali_num_threads=args.dali_num_threads,
                                        dali_py_num_workers=args.dali_py_num_workers,
                                        batch_size=args.batch_size,
                                        input_size=args.input_size,
                                        short_side_size=args.short_side_size,
                                        sequence_length=args.num_frames,
                                        use_rgb=args.use_rgb,
                                        mean=args.mean,
                                        std=args.std,
                                        mode="val",
                                        seed=1024)
    args.num_val_steps_per_epoch = len(data_loader_val)
    print("create data loader end")


    print("create model start")
    # base model export feature
    if args.model_name == "umt":
        from timm.models import create_model
        import video_models.umt
        base_model = create_model(
            args.model,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_name == "videomae_v1":
        from timm.models import create_model
        import video_models.videomae_v1
        base_model = create_model(
            args.model,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_name == "videomae_v2":
        from timm.models import create_model
        import video_models.videomae_v2
        base_model = create_model(
            args.model,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_name == "vswift":
        from timm.models import create_model
        import video_models.vswift
        base_model = create_model(
            args.model,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_name == "viclip":
        from timm.models import create_model
        import video_models.viclip
        base_model = create_model(
            args.model,
            input_resolution=args.input_size,
            pretrained=False,
            kernel_size=args.tubelet_size,
            center=True, 
            num_frames=args.num_frames,
            drop_path=0.0, 
            checkpoint_num=0,
            dropout=0.0)
        base_model.forward= base_model.forward_features_attentive_probe
    else:
        raise RuntimeError
    base_model = load_finetune_checkpoint(args, base_model)
    base_model.to(device)
    for name, p in base_model.named_parameters():
        p.requires_grad = False
    forward_base_model = torch.nn.DataParallel(base_model, device_ids=[args.local_rank])
    forward_base_model.eval()


    # attentive_probing
    attentive_probe_model = AttentionPoolingBlock(
                                        dim=args.embedding_size, 
                                        num_heads=args.default_attentive_head, 
                                        qkv_bias=True, 
                                        qk_scale=None,
                                        drop=0.0, 
                                        attn_drop=0.0, 
                                        drop_path=0.0, 
                                        norm_layer=partial(nn.LayerNorm, eps=1e-5), 
                                        out_dim=args.default_attentive_out_dim)
    ap_model = CustomModel(attentive_probe_model,
                        attentive_dim=args.default_attentive_out_dim,
                        num_classes=args.num_classes)
    print("create model end")


    best_lr, max_acc_top1, max_acc_top5 = 0, 0, 0
    for lr in args.default_lr_list:
        acc_top1, acc_top5 = find_peak(args, lr, ap_model, device,
                                       forward_base_model, data_loader_train, data_loader_val)
        if max_acc_top1 < acc_top1:
            best_lr, max_acc_top1, max_acc_top5 = lr, acc_top1, acc_top5

    
    print("best_lr: ", best_lr, "max_acc_top1: ", max_acc_top1, "max_acc_top5: ", max_acc_top5)
    if args.global_rank == 0:
        head_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format("dataset", "seed", 
                                                                                "default_epoch", "default_warmup_epochs",
                                                                                "default_weight_decay", "default_min_lr",
                                                                                "best_lr", "epoch", "max_acc_top1", "max_acc_top5")
        cur_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format(args.data_set, args.seed, 
                                                                            args.default_epoch, args.default_warmup_epochs, 
                                                                            args.default_weight_decay, args.default_min_lr, 
                                                                            best_lr, args.default_epoch, max_acc_top1, max_acc_top5)
        with open(args.report_txt_path, "a+") as writer:
            writer.write(head_info)
            writer.write(cur_info)