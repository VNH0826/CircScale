import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm.auto import tqdm
import pickle
import pandas as pd

from config import Config
from data.dataset import CircScaleDataset
from data.dataloader import create_dataloaders
from models.circscale import CircScale
from utils.metrics import mse_loss
from utils.visualization import plot_training_curves, save_visualization_metadata
from utils.helpers import set_seed, get_timestamp, save_model, save_losses, move_batch_to_device, empty_cuda_cache, \
    reset_tqdm


def get_task_config(metric):
    if metric.lower() == "delay":
        return {"hidden_dim": 32, "task_type": "delay"}
    elif metric.lower() == "area":
        return {"hidden_dim": 32, "task_type": "area"}
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def train_epoch(model, dataloader, optimizer, criterion, device):
    """train for one epoch"""
    model.train()
    epoch_loss = 0.0

    with tqdm(dataloader, desc="Training", position=0, leave=True) as pbar:
        for batch in pbar:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            pred_trajectory = model(batch)
            loss = criterion(pred_trajectory, batch['qor_trajectory'])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return epoch_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    val_loss = 0.0

    with tqdm(dataloader, desc="Validation", position=0, leave=True) as pbar:
        with torch.no_grad():
            for batch in pbar:
                batch = move_batch_to_device(batch, device)
                pred_trajectory = model(batch)
                loss = criterion(pred_trajectory, batch['qor_trajectory'])
                val_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.6f}")

    return val_loss / len(dataloader)


def train(args):
    """Main training function"""
    print(f"Setting dataset to: {args.dataset}")

    is_available, message = Config.validate_dataset(args.dataset, args.metric)
    if not is_available:
        print(f"Error: {message}")
        print("Available datasets:")
        for dataset_info in Config.list_datasets():
            status = "Available" if dataset_info['available'] else "Not Available"
            print(f"  - {dataset_info['name']}: {status}")
        return

    Config.set_dataset(args.dataset, args.metric)
    current_paths = Config.get_data_paths()

    print(f"Using dataset: {args.dataset}")
    print(f"CSV path: {current_paths['csv_path']}")
    print(f"Graph directory: {current_paths['graph_dir']}")

    task_config = get_task_config(args.metric)
    print(f"Using metric: {args.metric}, Task config: {task_config}")

    Config.create_dirs()
    set_seed(args.seed)

    device = Config.get_device()
    print(f"Using device: {device}")

    timestamp = get_timestamp()

    print("Loading dataset...")
    dataset = CircScaleDataset(Config.CSV_PATH(), Config.GRAPH_DIR(), train=True, metric_type=args.metric)

    if args.inductive:
        with open(args.test_designs, 'r') as f:
            test_designs = [line.strip() for line in f.readlines()]
        train_loader, val_loader = create_inductive_dataloaders(
            dataset, Config.BATCH_SIZE, test_designs)
    else:
        train_loader, val_loader = create_dataloaders(
            dataset, Config.BATCH_SIZE)

    sample_batch = next(iter(train_loader))
    input_node_dim = sample_batch['graph'].x.size(1)

    print("Creating model...")
    print(f"Configuration: MultiScale={args.use_multiscale}, DynamicInteraction={args.use_dynamic_interaction}")

    model = CircScale(
        input_node_dim=input_node_dim,
        hidden_dim=task_config["hidden_dim"],
        task_type=task_config["task_type"],
        num_heuristics=len(dataset.heuristic_to_idx),
        nhead=Config.NUM_HEADS,
        dim_feedforward=Config.TRANSFORMER_DIM * 4,
        dropout=Config.DROPOUT,
        num_layers=Config.NUM_LAYERS,
        use_multiscale=args.use_multiscale,
        max_hop=args.max_hop,
        use_dynamic_interaction=args.use_dynamic_interaction
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    criterion = mse_loss

    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    saved_model_version = None

    for epoch in range(Config.NUM_EPOCHS):
        reset_tqdm()

        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        empty_cuda_cache()

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        empty_cuda_cache()

        tqdm.write(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, "
                   f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_path = os.path.join(Config.SAVE_DIR(), f"best_model_{args.dataset}_{args.metric}_{epoch + 1}.pt")
            save_model(model, optimizer, epoch, val_loss, save_path)
            tqdm.write(f"Saved best model to {save_path}")
        else:
            patience_counter += 1

        save_path = os.path.join(Config.SAVE_DIR(), f"model_{args.dataset}_{args.metric}_{epoch + 1}.pt")
        save_model(model, optimizer, epoch, val_loss, save_path)

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            tqdm.write(f"Early stopping after {epoch + 1} epochs")
            break

    losses_path = os.path.join(Config.LOGS_DIR(), f"losses_{args.dataset}_{args.metric}_{timestamp}.pkl")
    save_losses(train_losses, val_losses, losses_path)

    plot_path = os.path.join(Config.VISUALIZATION_DIR(), f"training_curves_{args.dataset}_{args.metric}_{timestamp}.png")
    plot_training_curves(train_losses, val_losses, plot_path)

    save_visualization_metadata(Config, timestamp, Config.VISUALIZATION_DIR())

    vocab_path = os.path.join(Config.SAVE_DIR(), f"heuristic_vocab_{args.dataset}_{args.metric}_{timestamp}.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(dataset.heuristic_to_idx, f)

    print(f"\nTraining completed. Results saved with timestamp: {timestamp}")
    print(f"Dataset used: {args.dataset}")
    print(f"Metric used: {args.metric}")
    if saved_model_version:
        print(f"Best model saved as: best_model_{args.dataset}_{args.metric}_{epoch + 1}.pt")
    print(f"Vocabulary saved as: heuristic_vocab_{args.dataset}_{args.metric}_{timestamp}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CircScale model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--inductive", action="store_true", help="Use inductive setup")
    parser.add_argument("--test_designs", type=str, default="test_designs.txt",
                        help="File with test design names for inductive setup")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device index to use")

    parser.add_argument("--dataset", type=str, choices=['original', 'epfl', 'openabc', 'iscas85', 'iscas89', 'opencores'],
                        default='original', help="Dataset to use for training")
    parser.add_argument("--metric", type=str, choices=['Delay', 'Area'], default='Delay', help="QoR metric name")
    parser.add_argument("--list-datasets", action="store_true", help="List all available datasets and exit")

    parser.add_argument("--use_multiscale", action="store_true", help="Use multi-scale feature fusion")
    parser.add_argument("--max_hop", type=int, default=3, help="Maximum hop for multi-scale fusion")
    parser.add_argument("--use_dynamic_interaction", action="store_true",
                        help="Use dynamic graph-sequence adaptive interaction")

    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:")
        for dataset_info in Config.list_datasets():
            status = "Available" if dataset_info['available'] else "Not Available"
            print(f"  - {dataset_info['name']}: {status}")
            print(f"    CSV: {dataset_info['csv_path']}")
            print(f"    Graph Dir: {dataset_info['graph_dir']}")
            if not dataset_info['available']:
                print(f"    Issue: {dataset_info['status']}")
            print()
        exit(0)

    args.metric = args.metric.lower().capitalize() 

    if torch.cuda.is_available():
        Config.CUDA_DEVICE = args.cuda_device

    train(args)