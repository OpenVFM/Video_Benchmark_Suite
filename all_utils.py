import torch
from torch import nn
import torch.nn.functional as F

import torch.distributed as dist
import os
import numpy as np
import random
from collections import OrderedDict
import time
from collections import OrderedDict
from collections import defaultdict
from collections import deque
import datetime


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def load_state_dict(model,
                    state_dict,
                    prefix='',
                    ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".
            format(model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def load_finetune_checkpoint(args, video_model):
    checkpoint = torch.load(args.finetune, map_location='cpu')
    print("Load ckpt from %s" % args.finetune)
    
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    
    
    if args.model_name == 'videomae_v1' or args.model_name == 'videomae_v2':
        # videomae check
        for old_key in list(checkpoint_model.keys()):
            if old_key.startswith('_orig_mod.'):
                print("if old_key.startswith('_orig_mod.'):")
                new_key = old_key[10:]
                checkpoint_model[new_key] = checkpoint_model.pop(old_key)


    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict


    if args.model_name == 'viclip':
        all_keys = list(checkpoint_model.keys())
        vision_dict = OrderedDict()
        tex_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('vision_encoder.'):
                vision_dict[key[15:]] = checkpoint_model[key]
            elif key.startswith('text_encoder.'):
                tex_dict[key[13:]] = checkpoint_model[key]
            else:
                continue
        checkpoint_model = vision_dict

    
    if 'pos_embed' in checkpoint_model:
        print("'pos_embed' in checkpoint_model")

    if args.model_name == 'viclip':
        def inflate_weight(weight_2d, time_dim, center=True):
            print('Init center: {center}')
            if center:
                weight_3d = torch.zeros(*weight_2d.shape)
                weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
                middle_idx = time_dim // 2
                weight_3d[:, :, middle_idx, :, :] = weight_2d
            else:
                weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
                weight_3d = weight_3d / time_dim
            return weight_3d

        state_dict = checkpoint_model
        state_dict_3d = video_model.state_dict()
        for k in state_dict.keys():
            if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
                if len(state_dict_3d[k].shape) <= 2:
                    print('Ignore: {k}')
                    continue
                print('Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
                time_dim = state_dict_3d[k].shape[2]
                state_dict[k] = inflate_weight(state_dict[k], time_dim, center=True)

        pos_embed_checkpoint = state_dict['positional_embedding']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = (args.input_size // args.patch_size) ** 2
        orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print('Pos_emb from {orig_size} to {new_size}')
            extra_tokens = pos_embed_checkpoint[:1]
            pos_tokens = pos_embed_checkpoint[1:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            state_dict['positional_embedding'] = new_pos_embed
        checkpoint_model = state_dict

    load_state_dict(video_model, checkpoint_model)
    
    return video_model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64,
                         device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def min(self):
        return min(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            min=self.min,
            value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

        self.step_time_start = 0
        self.init = False
        self.tic = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, 
                        world_size=None, batch_size=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f} ({min:.4f} -- {max:.4f})')
        data_time = SmoothedValue(fmt='{avg:.4f} ({min:.4f} -- {max:.4f})')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]', 
            'eta: {eta}', 
            '{meters}',
            'time: {time}', 
            'data: {data}',
            'max mem: {memory:.0f}']

        if (world_size is not None) and (batch_size is not None):
            log_msg.append('video/s/gpu: {qps_v1}')
            log_msg.append('video/s: {qps_v2}')

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                if self.init:
                    if (world_size is not None) and (batch_size is not None):
                        try:
                            speed = print_freq * batch_size / (time.time() - self.tic)
                            self.tic = time.time()
                            speed_total = speed * world_size
                        except ZeroDivisionError:
                            speed = float("inf")
                            speed_total = float("inf")

                    eta_seconds = iter_time.global_avg * (len(iterable) - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                    if (world_size is not None) and (batch_size is not None):
                        speed = "{:.4f}".format(speed)
                        speed_total = "{:.4f}".format(speed_total)

                        print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),
                            time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB,
                            qps_v1=str(speed), qps_v2=str(speed_total)))
                    else:
                        print(
                            log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),
                                time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB))

                else:
                    self.init = True
                    self.tic = time.time()
                    self.step_time_start = time.time()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))