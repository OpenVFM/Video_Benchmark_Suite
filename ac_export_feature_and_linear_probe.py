import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import torch
from torch import distributed
import warnings
import argparse
from ac_lp_dataloader_dali import dali_dataloader
from all_utils import (setup_seed, setup_for_distributed, load_finetune_checkpoint)
import torch.nn.functional as F


class LogisticRegressionGPU(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.head = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.head(x)
        return x


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is torch.nn.Linear:
            l2_loss.append((module.weight**2).sum() / 2.0)
    return l2_alpha * torch.sqrt(sum(l2_loss))


@torch.no_grad()
def metric_acc(probs, gts, topk=(1,)):
    maxk = max(topk)
    batch_size = gts.size(0)
    _, pred = probs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(gts.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res[0]


def train_lbfgs(args, 
                train_feature, train_label, 
                val_feature, val_label, 
                c_weight):
    model_lr_gpu = LogisticRegressionGPU(args.embedding_size, args.num_classes)
    model_lr_gpu.train()
    model_lr_gpu.cuda()
    cross_entropy = torch.nn.CrossEntropyLoss()
    cross_entropy.cuda()
    lbfgs = torch.optim.LBFGS(model_lr_gpu.parameters(), max_iter=1000,)

    def closure():
        lbfgs.zero_grad()
        chunk_size = 10240
        for i in range(len(train_feature) // chunk_size + 1):
            end = min(len(train_feature), i * chunk_size + chunk_size)
            predict = model_lr_gpu(train_feature[i * chunk_size: end])
            ce_loss = cross_entropy(predict, train_label[i * chunk_size: end])
            l2loss = l2_regularization(model_lr_gpu, c_weight)
            loss = ce_loss + l2loss
            loss.backward()
        return loss
    
    if args.debug_flag:
        lbfgs.step(closure)
    else:
        try:
            lbfgs.step(closure)
        except:
            print("except RuntimeError")
            return 0

    model_lr_gpu.eval()
    x_predict = model_lr_gpu(val_feature)
    if torch.any(torch.isnan(x_predict)):
        print("torch.isnan")
        return 0

    with torch.no_grad():
        score = metric_acc(x_predict.cpu(), val_label.cpu())
    return score


def get_args():
    parser = argparse.ArgumentParser('Extract features using the videomae model', add_help=False)
    parser.add_argument('--data_root_path', default="fewshot_video/ActionRecognition")
    parser.add_argument('--data_csv_path', default="fewshot_video/ActionRecognition")
    parser.add_argument('--save_report', default="fewshot_video_report/ActionRecognition")

    parser.add_argument('--data_set', default="SSV2")
    parser.add_argument('--num_shots', default=10)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_step", default=8, type=int)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--model_name', default='viclip')
    parser.add_argument('--model', default='clip_joint_b16')
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--finetune', default='checkpoint/viclip/viclip-B-16____InternVid_10MFLT.pt')
    parser.add_argument("--num_frames", default=8, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--tubelet_size", default=1, type=int)
    parser.add_argument("--embedding_size", default=512, type=int)

    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--mean', nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--normalize', default=False)

    # default
    parser.add_argument('--dali_num_threads', default=8, type=int)
    parser.add_argument('--dali_py_num_workers', default=14, type=int)
    parser.add_argument('--use_rgb', default=False)
    parser.add_argument('--debug_flag', default=False)
    return parser.parse_args()


def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_model(args):
    print("create model start")
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
        base_model.forward= base_model.forward_features
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
        base_model.forward= base_model.forward_features
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
        base_model.forward= base_model.forward_features
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
        base_model.forward= base_model.forward_features
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
        base_model.forward= base_model.forward_features
    else:
        raise RuntimeError
    return base_model


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
        report_txt_path = os.path.join(args.save_report, args.data_set, "report_linear_probe_{}.txt".format(args.model_name))
        head_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format("model_name", "model", "num_frames", 
                                                                      "input_size", "tubelet_size", "finetune")
        cur_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<30}\n".format(args.model_name, args.model, 
                                                 args.num_frames, args.input_size, args.tubelet_size, args.finetune)
        with open(report_txt_path, "a+") as writer:
            writer.write(head_info)
            writer.write(cur_info)


    print("create data loader start")
    data_loader_train = dali_dataloader(args.data_root_path,
                                        args.data_csv_path,
                                        args.data_set,
                                        num_shots=args.num_shots,
                                        dali_num_threads=args.dali_num_threads,
                                        dali_py_num_workers=args.dali_py_num_workers,
                                        batch_size=args.batch_size,
                                        input_size=args.input_size,
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
                                        num_shots=args.num_shots,
                                        dali_num_threads=args.dali_num_threads,
                                        dali_py_num_workers=args.dali_py_num_workers,
                                        batch_size=args.batch_size,
                                        input_size=args.input_size,
                                        sequence_length=args.num_frames,
                                        use_rgb=args.use_rgb,
                                        mean=args.mean,
                                        std=args.std,
                                        mode="val",
                                        seed=1024)
    args.num_val_steps_per_epoch = len(data_loader_val)
    print("create data loader end")


    base_model = get_model(args)
    base_model = load_finetune_checkpoint(args, base_model)
    base_model.to(device)
    forward_base_model = torch.nn.DataParallel(base_model, device_ids=[args.local_rank])


    # base model export feature
    print("export feature start")
    print("export train feature")
    with torch.no_grad():
        train_feature_list = []
        train_label_list = []
        for step_idx, (samples, targets) in enumerate(data_loader_train):
            print(step_idx, "/", args.num_train_steps_per_epoch)
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets = targets.view(-1)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = forward_base_model(samples)            
                if args.normalize:
                    outputs = F.normalize(outputs, dim=-1)
            outputs = outputs.float()
            
            train_feature_list.append(outputs)
            train_label_list.append(targets)
        train_feature = torch.cat(train_feature_list, axis=0)
        train_label = torch.cat(train_label_list, axis=0)

    print("export val feature")
    with torch.no_grad():
        val_feature_list = []
        val_label_list = []
        for step_idx, (samples, targets) in enumerate(data_loader_val):
            print(step_idx, "/", args.num_val_steps_per_epoch)
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets = targets.view(-1)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = forward_base_model(samples)
                if args.normalize:
                    outputs = F.normalize(outputs, dim=-1)
            outputs = outputs.float()

            val_feature_list.append(outputs)
            val_label_list.append(targets)
        val_feature = torch.cat(val_feature_list, axis=0)
        val_label = torch.cat(val_label_list, axis=0)
    print("export feature end")


    # gather the train-feature from all processes
    _gather_train_feature = [
        torch.zeros_like(train_feature).float().cuda() for _ in range(args.world_size)
    ]
    _gather_train_label = [
        torch.zeros_like(train_label).cuda() for _ in range(args.world_size)
    ]
    distributed.all_gather(_gather_train_feature, train_feature)
    merger_train_feature = torch.cat(_gather_train_feature)
    distributed.all_gather(_gather_train_label, train_label)
    merger_train_label = torch.cat(_gather_train_label)

    # gather the val-feature from all processes
    _gather_val_feature = [
        torch.zeros_like(val_feature).float().cuda() for _ in range(args.world_size)
    ]
    _gather_val_label = [
        torch.zeros_like(val_label).cuda() for _ in range(args.world_size)
    ]
    distributed.all_gather(_gather_val_feature, val_feature)
    merger_val_feature = torch.cat(_gather_val_feature)
    distributed.all_gather(_gather_val_label, val_label)
    merger_val_label = torch.cat(_gather_val_label)


    if args.global_rank == 0:
        print("start lbfgs")
        search_list = [10**-14, 10**-12, 10**-10, 10**-8, 10**-6, 10**-4, 10**-2, 1, 10**2, 10**4, 10**6]
        acc_list = []
        for c_weight in search_list:
            acc_val = train_lbfgs(args, merger_train_feature, merger_train_label,
                                merger_val_feature, merger_val_label, c_weight)
            print("cur c_weight: ", c_weight, "cur acc_val: ", acc_val)
            acc_list.append(acc_val)

        max_idx = np.argmax(acc_list)
        bast_c_weight = search_list[max_idx]
        best_acc = acc_list[max_idx]
        print("bast_c_weight: ", bast_c_weight, "best_acc: ", best_acc)

        head_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<20}\t{:<20}\n".format("dataset", "seed", "step", 
                                                                                "fewshot", "c_weight", "acc")
        cur_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<20}\t{:<20}\n".format(args.data_set, args.seed, 
                                                    -1, args.num_shots, 
                                                    bast_c_weight, best_acc)
        with open(report_txt_path, "a+") as writer:
            writer.write(head_info)
            writer.write(cur_info)

        # binary_search
        c_left = 1e-1 * bast_c_weight
        c_right = 1e1 * bast_c_weight
        c_final = None
        acc_final = None
        for step in range(args.num_step):
            acc_left = train_lbfgs(args, merger_train_feature, merger_train_label, 
                                merger_val_feature, merger_val_label, c_left)            
            acc_right = train_lbfgs(args, merger_train_feature, merger_train_label, 
                                merger_val_feature, merger_val_label, c_right)

            # np.log10(100)=2
            # np.power(10, 2)=100
            if acc_left < acc_right:
                c_final = c_right
                acc_final = acc_right
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                c_final = c_left
                acc_final = acc_left
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)      

            c_left = np.power(10, c_left)
            c_right = np.power(10, c_right)

            print("bast_c_weight: ", c_final, "best_acc: ", acc_final)

            head_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<20}\t{:<20}\n".format("dataset", "seed", "step", 
                                                                                    "fewshot", "c_weight", "acc")
            cur_info = "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<20}\t{:<20}\n".format(args.data_set, args.seed, 
                                                        step, args.num_shots, 
                                                        c_final, acc_final)
            with open(report_txt_path, "a+") as writer:
                writer.write(head_info)
                writer.write(cur_info)