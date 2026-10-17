"""Paper-first PyTorch port of cruiseresearchgroup/TSCP2 and keras-tcn.

The standalone worker trains and scores the fixed network. The MIT notice for
keras-tcn is retained under reference_sources/tscp2/keras-tcn/LICENSE.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# NumPy MKL and Torch bundle different OpenMP DLLs in some Windows envs.
# Sequential MKL avoids loading a second runtime; Torch still owns its threads.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
# Load Torch's native runtime before NumPy/MKL on Windows.
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

# File import keeps the worker independent of the parent interpreter/package.
_spec = importlib.util.spec_from_file_location("tscp2_adapter", Path(__file__).resolve().parents[1] / "tscp2_adapter.py")
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)


class KerasBatchNorm(nn.Module):
    """Keras defaults: eps=.001, momentum=.99, biased moving variance."""
    def __init__(self, channels, update_statistics=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.update_statistics = update_statistics

    def forward(self, x):
        if self.training and self.update_statistics:
            next_mean, next_variance = self.running_mean.clone(), self.running_var.clone()
            output = F.batch_norm(x, next_mean, next_variance,
                                  self.weight, self.bias, True, .01, .001)
            # Fused BatchNorm normalizes by biased variance but updates running
            # variance with an unbiased estimate. Undo only that Bessel factor
            # to reproduce Keras moving variance, without many CUDA kernels.
            count = x.numel() // x.shape[1]
            with torch.no_grad():
                self.running_mean.copy_(next_mean)
                self.running_var.mul_(.99/count).add_(next_variance, alpha=1.-1./count)
            return output
        else:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, .01, .001)


class CausalConv(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, (self.dilation[0] * (self.kernel_size[0]-1), 0)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, filters, kernel, dilation, update_statistics):
        super().__init__()
        self.conv1 = CausalConv(channels, filters, kernel, dilation=dilation)
        self.norm1 = KerasBatchNorm(filters, update_statistics)
        self.conv2 = CausalConv(filters, filters, kernel, dilation=dilation)
        self.norm2 = KerasBatchNorm(filters, update_statistics)
        self.match = nn.Conv1d(channels, filters, 1) if channels != filters else nn.Identity()

    def forward(self, x):
        skip = F.relu(self.norm1(self.conv1(x)))
        skip = F.relu(self.norm2(self.conv2(skip)))
        return F.relu(self.match(x) + skip), skip


class TSCP2Encoder(nn.Module):
    def __init__(self, window, config):
        super().__init__()
        self.window = window
        blocks, channels = [], 1
        for _ in range(config["stacks"]):
            for dilation in config["dilations"]:
                blocks.append(ResidualBlock(channels, config["filters"], config["kernel_size"], dilation,
                                            config["batch_norm_update_statistics"]))
                channels = config["filters"]
        self.blocks = nn.ModuleList(blocks)
        half = window // 2
        self.projection = nn.Sequential(nn.Linear(window * channels, 2*half), nn.ReLU(),
                                        nn.Linear(2*half, half), nn.ReLU(),
                                        nn.Linear(half, config["code_size"]),
                                        KerasBatchNorm(config["code_size"], config["batch_norm_update_statistics"]))
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.normal_(layer.weight, mean=0., std=.05)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def temporal_features(self, x):
        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)
        return torch.stack(skips).sum(0)

    def forward(self, x):
        # Source Flatten consumes (batch, time, channels), not channels first.
        x = self.temporal_features(x.transpose(1, 2)).transpose(1, 2).flatten(1)
        return self.projection(x)


def info_nce(history, future, temperature):
    # TensorFlow l2_normalize uses sqrt(max(sum(x**2), epsilon=1e-12)).
    history = F.normalize(history, dim=1, eps=1e-6)
    future = F.normalize(future, dim=1, eps=1e-6)
    similarity = history @ future.T
    loss = F.cross_entropy(similarity / temperature,
                           torch.arange(len(history), device=history.device), reduction="sum")
    positive = similarity.diag().mean()
    negative = (similarity.sum() - similarity.diag().sum()) / (len(history)*(len(history)-1))
    return loss, positive, negative


class KerasAdam(torch.optim.Adam):
    """TF 2.2 Adam epsilon-hat placement, expressed with PyTorch Adam.

    TF: lr*sqrt(1-beta2**t)/(1-beta1**t)*m/(sqrt(v)+epsilon_hat).
    Scaling Torch's epsilon by 1/sqrt(1-beta2**t) gives that exact formula.
    Floating-point kernels and framework RNG are not claimed bit-identical.
    """
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr=lr, betas=(.9,.999), eps=1e-7)
        self.iterations = 0

    def step(self, closure=None):
        self.iterations += 1
        for group in self.param_groups:
            group["eps"] = 1e-7 / (1-group["betas"][1]**self.iterations)**.5
        return super().step(closure)


def initialize(seed, device_name):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("TSCP2_CPU_THREADS", "4")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(device_name)


def train_encoder(values, train_stops, window, config, output, device):
    pairs, locations = _adapter.training_pairs(values, train_stops, window, config["training_stride"],
                                               last_start_inclusive=False)
    batches = _adapter.separated_batches(locations, config["batch_size"], 2*window, config["seed"])
    np.savez_compressed(output / "training_samples.npz", locations=locations, batches=batches)
    data = torch.as_tensor(pairs, device=device).unsqueeze(-1)
    batch_indices = torch.as_tensor(batches, device=device)
    model = TSCP2Encoder(window, config).to(device)
    optimizer = KerasAdam(model.parameters(), lr=config["learning_rate"])
    history, stalled = [], 0
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "positive_similarity", "negative_similarity", "seconds", "stalled"])
        writer.writeheader()
        stream.flush()
        for epoch in range(config["max_epochs"]):
            started = time.perf_counter()
            totals = torch.zeros(3, device=device)
            model.train()
            for indices in batch_indices:
                batch = data[indices]
                optimizer.zero_grad(set_to_none=True)
                loss, positive, negative = info_nce(model(batch[:, :window]), model(batch[:, window:]), config["temperature"])
                loss.backward()
                optimizer.step()
                totals += torch.stack([loss.detach(), positive.detach(), negative.detach()])
            averages = (totals / len(batches)).cpu().tolist()
            if not np.isfinite(averages).all():
                raise FloatingPointError("TSCP2 training loss/similarity is not finite")
            if epoch > 5 and (abs(averages[0]-history[-1]["loss"]) < .0001 or history[-1]["loss"] < averages[0]):
                stalled += 1
            else:
                stalled = 0
            row = dict(epoch=epoch+1, loss=averages[0], positive_similarity=averages[1],
                       negative_similarity=averages[2], seconds=time.perf_counter()-started, stalled=stalled)
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print("window={} epoch={} loss={:.6f} positive={:.4f} negative={:.4f} seconds={:.1f} stalled={}".format(
                window, epoch+1, *averages, row["seconds"], stalled), flush=True)
            if stalled == 4:
                break
    model.eval()
    torch.save(dict(window=window, parameters=config, state_dict=model.state_dict(),
                    epochs=len(history), batch_count=len(batches)), output / "encoder.pt")
    return model, dict(epochs=len(history), training_pairs=len(pairs), retained_pairs=int(batches.size),
                       batch_count=len(batches), stopping_reason="source_patience_four" if stalled == 4 else "max_epochs",
                       final_loss=history[-1]["loss"])


def score_encoder(model, values, window, device, step=1):
    rows = []
    with torch.inference_mode():
        for values_row in values:
            # Encode each window once; shared encoder gives identical embeddings.
            windows = np.lib.stride_tricks.sliding_window_view(values_row, window).copy()
            codes = []
            for start in range(0, len(windows), 512):
                batch = torch.as_tensor(windows[start:start+512], dtype=torch.float32, device=device).unsqueeze(-1)
                codes.append(F.normalize(model(batch), dim=1, eps=1e-6))
            codes = torch.cat(codes)
            similarity = np.full(len(values_row), np.nan)
            similarity[window:len(values_row)-window+1] = (codes[:-window] * codes[window:]).sum(1).cpu().numpy()
            if step != 1:
                selected = np.zeros(len(values_row), dtype=bool)
                # Original Yahoo extract_windows excludes the last equal start.
                selected[window:len(values_row)-window:step] = True
                similarity[~selected] = np.nan
            rows.append(similarity)
    return np.asarray(rows)


def worker(input_path, config_path, output, window, device_name):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    device = initialize(config["seed"], device_name)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("worker output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(input_path, allow_pickle=False) as data:
        values, train_stops, validation_stops = data["values"], data["train_stops"], data["validation_stops"]
    model, training = train_encoder(values, train_stops, window, config, output, device)
    step = config["inference_stride"]
    similarity = score_encoder(model, values, window, device, step)
    scores = np.asarray([_adapter.similarity_drop(row, window, step) for row in similarity])
    validation_peaks = np.asarray([_adapter.phase_peak_scores(row, window, stop, step) for row, stop in zip(scores, validation_stops)])
    test_peaks = np.asarray([_adapter.phase_peak_scores(row, window, len(row), step) for row in scores])
    np.savez_compressed(output / "scores.npz", similarity=similarity, scores=scores,
                        validation_peaks=validation_peaks, test_peaks=test_peaks)
    metadata = dict(training, device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                    torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__, seed=config["seed"],
                    parameter_count=sum(p.numel() for p in model.parameters()), framework="PyTorch paper-first source port")
    (output / "worker.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    worker(args.input, args.config, args.output, args.window, args.device)
