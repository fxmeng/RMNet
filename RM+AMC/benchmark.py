import argparse
import os
import sys
import csv
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from timm.models import is_model, list_models
from timm.data import resolve_data_config
from timm.utils import setup_default_logging
from models.rmnet import *
import thop

has_apex = False
try:
  from apex import amp
  has_apex = True
except ImportError:
  pass

has_native_amp = False
try:
  if getattr(torch.cuda.amp, 'autocast') is not None:
    has_native_amp = True
except AttributeError:
  pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='PyTorch Benchmark')

# benchmark specific args
parser.add_argument('--model-list',
                    metavar='NAME',
                    default='',
                    help='txt file based list of model names to benchmark')
parser.add_argument(
    '--bench',
    default='both',
    type=str,
    help=
    "Benchmark mode. One of 'inference', 'train', 'both'. Defaults to 'inference'"
)
parser.add_argument(
    '--detail',
    action='store_true',
    default=False,
    help='Provide train fwd/bwd/opt breakdown detail if True. Defaults to False'
)
parser.add_argument('--results-file',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--num-warm-iter',
                    default=50,
                    type=int,
                    metavar='N',
                    help='Number of warmup iterations (default: 10)')
parser.add_argument('--num-bench-iter',
                    default=50,
                    type=int,
                    metavar='N',
                    help='Number of benchmark iterations (default: 40)')

# common inference / train args
parser.add_argument('--model',
                    '-m',
                    metavar='NAME',
                    default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to fine tune')
parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--img-size',
                    default=None,
                    type=int,
                    metavar='N',
                    help='Input image dimension, uses model default if empty')
parser.add_argument(
    '--input-size',
    default=(3,32,32),
    nargs=3,
    type=int,
    metavar='N N N',
    help=
    'Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty'
)
parser.add_argument('--num-classes',
                    type=int,
                    default=10,
                    help='Number classes in dataset')
parser.add_argument(
    '--gp',
    default=None,
    type=str,
    metavar='POOL',
    help=
    'Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.'
)
parser.add_argument('--channels-last',
                    action='store_true',
                    default=False,
                    help='Use channels_last memory layout')
parser.add_argument(
    '--amp',
    action='store_true',
    default=False,
    help=
    'use PyTorch Native AMP for mixed precision training. Overrides --precision arg.'
)
parser.add_argument(
    '--precision',
    default='float32',
    type=str,
    help='Numeric precision. One of (amp, float32, float16, bfloat16, tf32)')
parser.add_argument('--torchscript',
                    dest='torchscript',
                    action='store_true',
                    help='convert model torchscript for inference')


def timestamp(sync=False):
  return time.perf_counter()


def cuda_timestamp(sync=False, device=None):
  if sync:
    torch.cuda.synchronize(device=device)
  return time.perf_counter()


def count_params(model: nn.Module):
  return sum([m.numel() for m in model.parameters()])


def resolve_precision(precision: str):
  assert precision in ('amp', 'float16', 'bfloat16', 'float32')
  use_amp = False
  model_dtype = torch.float32
  data_dtype = torch.float32
  if precision == 'amp':
    use_amp = True
  elif precision == 'float16':
    model_dtype = torch.float16
    data_dtype = torch.float16
  elif precision == 'bfloat16':
    model_dtype = torch.bfloat16
    data_dtype = torch.bfloat16
  return use_amp, model_dtype, data_dtype


class BenchmarkRunner:

  def __init__(self,
               model_name,
               detail=False,
               device='cuda',
               torchscript=False,
               precision='float32',
               num_warm_iter=10,
               num_bench_iter=50,
               **kwargs):
    self.model_name = model_name
    self.detail = detail
    self.device = device
    self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(
        precision)
    self.channels_last = kwargs.pop('channels_last', False)
    self.amp_autocast = torch.cuda.amp.autocast if self.use_amp else suppress
    self.num_classes = kwargs.pop('num_classes')
    self.ckpt_path = kwargs.pop('ckpt_path')
    self.model = ckpt_to_model(self.ckpt_path,self.model_name,int(self.num_classes))
    #self.model = torch.load(model_name)
    print(self.model)
    

    self.model.to(
        device=self.device,
        dtype=self.model_dtype,
        memory_format=torch.channels_last if self.channels_last else None)
    self.param_count = count_params(self.model)
    _logger.info('Model %s created, param count: %d' %
                 (model_name, self.param_count))
    if torchscript:
      self.model = torch.jit.script(self.model)

    data_config = resolve_data_config(kwargs,
                                      model=self.model,
                                      use_test_size=True)
    self.input_size = data_config['input_size']
    self.batch_size = kwargs.pop('batch_size', 256)
    print(thop.profile(self.model,(torch.randn(1,3,self.input_size[1],self.input_size[2]).to(self.device),)))

    self.example_inputs = None
    self.num_warm_iter = num_warm_iter
    self.num_bench_iter = num_bench_iter
    self.log_freq = num_bench_iter // 5
    if 'cuda' in self.device:
      self.time_fn = partial(cuda_timestamp, device=self.device)
    else:
      self.time_fn = timestamp

  def _init_input(self):
    self.example_inputs = torch.randn((self.batch_size,) + self.input_size,
                                      device=self.device,
                                      dtype=self.data_dtype)
    if self.channels_last:
      self.example_inputs = self.example_inputs.contiguous(
          memory_format=torch.channels_last)


class InferenceBenchmarkRunner(BenchmarkRunner):

  def __init__(self, model_name, device='cuda', torchscript=False, **kwargs):
    super().__init__(model_name=model_name,
                     device=device,
                     torchscript=torchscript,
                     **kwargs)
    self.model.eval()

  def run(self):

    def _step():
      t_step_start = self.time_fn()
      with self.amp_autocast():
        output = self.model(self.example_inputs)
      t_step_end = self.time_fn(True)
      return t_step_end - t_step_start

    _logger.info(
        f'Running inference benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
        f'input size {self.input_size} and batch size {self.batch_size}.')

    with torch.no_grad():
      self._init_input()

      for _ in range(self.num_warm_iter):
        _step()

      total_step = 0.
      num_samples = 0
      t_run_start = self.time_fn()
      for i in range(self.num_bench_iter):
        delta_fwd = _step()
        total_step += delta_fwd
        num_samples += self.batch_size
        num_steps = i + 1
        if num_steps % self.log_freq == 0:
          _logger.info(f"Infer [{num_steps}/{self.num_bench_iter}]."
                       f" {num_samples / total_step:0.2f} samples/sec."
                       f" {1000 * total_step / num_steps:0.3f} ms/step.")
      t_run_end = self.time_fn(True)
      t_run_elapsed = t_run_end - t_run_start

    results = dict(
        samples_per_sec=round(num_samples / t_run_elapsed, 2),
        step_time=round(1000 * total_step / self.num_bench_iter, 3),
        batch_size=self.batch_size,
        img_size=self.input_size[-1],
        param_count=round(self.param_count / 1e6, 2),
    )

    _logger.info(
        f"Inference benchmark of {self.model_name} done. "
        f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step"
    )

    return results


def decay_batch_exp(batch_size, factor=0.5, divisor=16):
  out_batch_size = batch_size * factor
  if out_batch_size > divisor:
    out_batch_size = (out_batch_size + 1) // divisor * divisor
  else:
    out_batch_size = batch_size - 1
  return max(0, int(out_batch_size))


def _try_run(model_name, bench_fn, initial_batch_size, bench_kwargs):
  batch_size = initial_batch_size
  results = dict()
  while batch_size >= 1:
    try:
      bench = bench_fn(model_name=model_name,
                       batch_size=batch_size,
                       **bench_kwargs)
      results = bench.run()
      return results
    except RuntimeError as e:
      torch.cuda.empty_cache()
      batch_size = decay_batch_exp(batch_size)
      print(
          f'Error: {str(e)} while running benchmark. Reducing batch size to {batch_size} for retry.'
      )
  return results


def benchmark(args):
  if args.amp:
    _logger.warning("Overriding precision to 'amp' since --amp flag set.")
    args.precision = 'amp'
  _logger.info(f'Benchmarking in {args.precision} precision. '
               f'{"NHWC" if args.channels_last else "NCHW"} layout. '
               f'torchscript {"enabled" if args.torchscript else "disabled"}')

  bench_kwargs = vars(args).copy()
  bench_kwargs.pop('amp')
  model = bench_kwargs.pop('model')
  batch_size = bench_kwargs.pop('batch_size')

  bench_fns = (InferenceBenchmarkRunner,)
  prefixes = ('infer',)
  model_results = OrderedDict(model=model)
  for prefix, bench_fn in zip(prefixes, bench_fns):
    run_results = _try_run(model,
                           bench_fn,
                           initial_batch_size=batch_size,
                           bench_kwargs=bench_kwargs)
    if prefix:
      run_results = {'_'.join([prefix, k]): v for k, v in run_results.items()}
    model_results.update(run_results)
  param_count = model_results.pop('infer_param_count',
                                  model_results.pop('train_param_count', 0))
  model_results.setdefault('param_count', param_count)
  model_results.pop('train_param_count', 0)
  return model_results


def main():
  setup_default_logging()
  args = parser.parse_args()
  model_cfgs = []
  model_names = []

  if args.model_list:
    args.model = ''
    with open(args.model_list) as f:
      model_names = [line.rstrip() for line in f]
    model_cfgs = [(n, None) for n in model_names]
  elif args.model == 'all':
    # validate all models in a list of names with pretrained checkpoints
    args.pretrained = True
    model_names = list_models(pretrained=True, exclude_filters=['*in21k'])
    model_cfgs = [(n, None) for n in model_names]
  elif not is_model(args.model):
    # model name doesn't exist, try as wildcard filter
    model_names = list_models(args.model)
    model_cfgs = [(n, None) for n in model_names]

  if len(model_cfgs):
    results_file = args.results_file or './benchmark.csv'
    _logger.info(
        'Running bulk validation on these pretrained models: {}'.format(
            ', '.join(model_names)))
    results = []
    try:
      for m, _ in model_cfgs:
        if not m:
          continue
        args.model = m
        r = benchmark(args)
        results.append(r)
    except KeyboardInterrupt as e:
      pass
    sort_key = 'train_samples_per_sec' if 'train' in args.bench else 'infer_samples_per_sec'
    results = sorted(results, key=lambda x: x[sort_key], reverse=True)
    if len(results):
      write_results(results_file, results)

    import json
    json_str = json.dumps(results, indent=4)
    print(json_str)
  else:
    benchmark(args)


def write_results(results_file, results):
  with open(results_file, mode='w') as cf:
    dw = csv.DictWriter(cf, fieldnames=results[0].keys())
    dw.writeheader()
    for r in results:
      dw.writerow(r)
    cf.flush()


if __name__ == '__main__':
  main()
