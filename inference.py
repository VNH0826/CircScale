# inference_baseline.py
import os
import torch
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from data.dataset import CircScaleDataset
from data.dataloader import create_dataloaders
from models.circscale import CircScale
from utils.metrics import mean_absolute_percentage_error
from utils.visualization import plot_training_curves, save_visualization_metadata
from utils.helpers import set_seed, get_timestamp, load_model, move_batch_to_device, empty_cuda_cache, reset_tqdm
from utils.visualization import plot_qor_trajectory


def get_task_config(metric):
    if metric.lower() == "delay":
        return {"hidden_dim": 32, "task_type": "delay"}
    elif metric.lower() == "area":
        return {"hidden_dim": 32, "task_type": "area"}
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def denormalize_qor(qor_normalized, mean, std):
    """Denormalize QoR values"""
    return qor_normalized * std + mean


def plot_mape_by_circuit_simple(design_mapes, design_names, overall_mape,
                                metric_name="QoR", dataset_name="Dataset",
                                save_path=None):
    sorted_data = sorted(zip(design_names, design_mapes), key=lambda x: x[0])
    sorted_design_names, sorted_design_mapes = zip(*sorted_data)

    plt.figure(figsize=(12, 8))

    bars = plt.bar(range(len(sorted_design_mapes)), sorted_design_mapes,
                   color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1)

    plt.xticks(range(len(sorted_design_names)), sorted_design_names, rotation=45, ha='right')
    plt.xlabel('Circuit Design', fontsize=12, fontweight='bold')
    plt.ylabel(f'MAPE (%) - {metric_name}', fontsize=12, fontweight='bold')

    plt.title(f'{dataset_name.upper()} Dataset - {metric_name} Prediction Accuracy',
              fontsize=14, fontweight='bold', pad=20)

    plt.axhline(y=overall_mape, color='red', linestyle='--', linewidth=2,
                label=f'Overall MAPE: {overall_mape:.2f}%', alpha=0.8)

    for bar, mape in zip(bars, sorted_design_mapes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{mape:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MAPE visualization saved to: {save_path}")

    plt.show()


def evaluate_model(model, dataloader, device, dataset, metric="Delay"):
    """Evaluate model on dataloader"""
    model.eval()

    all_preds = []
    all_targets = []
    all_design_names = []
    all_trajectories_pred = []
    all_trajectories_true = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch_device = move_batch_to_device(batch, device)

            # Forward pass
            pred_trajectory = model(batch_device)

            # Store predictions and targets
            for i in range(len(batch['design_name'])):
                design_name = batch['design_name'][i]
                pred_traj = pred_trajectory[i].cpu().numpy()
                true_traj = batch['qor_trajectory'][i].cpu().numpy()

                # Store final QoR
                fin_p = pred_traj[-1]
                fin_t = true_traj[-1]

                all_preds.append(fin_p)
                all_targets.append(fin_t)
                all_design_names.append(design_name)

                # Store trajectories
                all_trajectories_pred.append(pred_traj)
                all_trajectories_true.append(true_traj)

    # Calculate MAPE normally
    mape = mean_absolute_percentage_error(all_targets, all_preds)

    # Calculate MAPE per circuit
    unique_designs = []
    design_mapes = []

    for design in set(all_design_names):
        indices = [i for i, d in enumerate(all_design_names) if d == design]
        design_preds = [all_preds[i] for i in indices]
        design_targets = [all_targets[i] for i in indices]

        design_mape = mean_absolute_percentage_error(design_targets, design_preds)
        unique_designs.append(design)
        design_mapes.append(design_mape)

    return {
        'mape': mape,
        'design_names': unique_designs,
        'design_mapes': design_mapes,
        'all_preds': all_preds,
        'all_targets': all_targets,
        'all_design_names': all_design_names,
        'all_trajectories_pred': all_trajectories_pred,
        'all_trajectories_true': all_trajectories_true
    }


def inference(args):
    """Main inference function"""
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

    # Create directories
    Config.create_dirs()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Get device
    device = Config.get_device()
    print(f"Using device: {device}")

    # Get task configuration
    task_config = get_task_config(args.metric)
    print(f"Using metric: {args.metric}, Task config: {task_config}")

    # Create timestamp for this run
    timestamp = get_timestamp()

    # Load heuristic vocabulary
    with open(args.vocab_path, 'rb') as f:
        heuristic_to_idx = pickle.load(f)

    # Load dataset
    print("Loading dataset...")
    dataset = CircScaleDataset(Config.CSV_PATH(), Config.GRAPH_DIR(), heuristic_to_idx, train=False, metric_type=args.metric)

    # Create dataloaders
    if args.inductive:
        # For inductive setup, we need to specify test designs
        with open(args.test_designs, 'r') as f:
            test_designs = [line.strip() for line in f.readlines()]

        _, test_loader = create_inductive_dataloaders(
            dataset, Config.BATCH_SIZE, test_designs)
    else:
        # Recipe-Inductive setup
        _, val_loader = create_dataloaders(dataset, Config.BATCH_SIZE)
        test_loader = val_loader  # 在Recipe-Inductive中，验证集就是测试集

    # Get input dimension from first graph
    sample_batch = next(iter(test_loader))
    input_node_dim = sample_batch['graph'].x.size(1)

    # Create model with task-specific configuration
    print("Creating model...")
    print(f"Configuration: MultiScale={args.use_multiscale}, DynamicInteraction={args.use_dynamic_interaction}")
    model = CircScale(
        input_node_dim=input_node_dim,
        hidden_dim=task_config["hidden_dim"],
        task_type=task_config["task_type"],
        num_heuristics=len(heuristic_to_idx),
        nhead=Config.NUM_HEADS,
        dim_feedforward=Config.TRANSFORMER_DIM * 4,
        dropout=Config.DROPOUT,
        num_layers=Config.NUM_LAYERS,
        use_multiscale=args.use_multiscale,
        max_hop=args.max_hop,
        use_dynamic_interaction=args.use_dynamic_interaction
    ).to(device)

    # Load model
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    model, _, _, _ = load_model(model, optimizer, args.model_path, device)

    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, dataset, args.metric)

    # Empty CUDA cache if needed
    empty_cuda_cache()

    # Print overall MAPE
    print(f"Overall MAPE: {results['mape']:.2f}%")

    # Save results
    results_path = os.path.join(Config.RESULTS_DIR(), f"results_{args.dataset}_{args.metric}_{timestamp}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    # Plot MAPE by circuit with Overall MAPE reference line
    mape_plot_path = os.path.join(Config.VISUALIZATION_DIR(), f"mape_by_circuit_{args.dataset}_{args.metric}_{timestamp}.png")
    plot_mape_by_circuit_simple(
        results['design_mapes'],
        results['design_names'],
        overall_mape=results['mape'],
        metric_name=args.metric,
        dataset_name=args.dataset,
        save_path=mape_plot_path
    )

    # Plot QoR trajectories for a few designs (if requested)
    if args.plot_trajectories:
        traj_dir = os.path.join(Config.VISUALIZATION_DIR(), "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        # Get unique designs
        unique_designs = list(set(results['all_design_names']))

        # Plot for a subset of designs
        for design in unique_designs[:min(5, len(unique_designs))]:
            # Find first occurrence of this design
            idx = results['all_design_names'].index(design)

            # Plot trajectory
            traj_path = os.path.join(traj_dir,
                                     f"trajectory_{design.replace('.bench', '')}_{args.dataset}_{args.metric}_{timestamp}.png")
            plot_qor_trajectory(
                results['all_trajectories_true'][idx],
                results['all_trajectories_pred'][idx],
                design,
                save_path=traj_path
            )

    # Create detailed CSV with trajectories
    detailed_data = []
    for i, design in enumerate(results['all_design_names']):
        row_data = {
            'design_name': design,
            'predicted_final_qor': results['all_preds'][i],
            'true_final_qor': results['all_targets'][i],
            'mape': abs((results['all_targets'][i] - results['all_preds'][i]) / results['all_targets'][i]) * 100
        }

        # Add trajectory data
        for step in range(len(results['all_trajectories_true'][i])):
            row_data[f'true_qor_step_{step + 1}'] = results['all_trajectories_true'][i][step]
            row_data[f'pred_qor_step_{step + 1}'] = results['all_trajectories_pred'][i][step]

        detailed_data.append(row_data)

    # Create and save detailed DataFrame
    detailed_df = pd.DataFrame(detailed_data)
    detailed_csv_path = os.path.join(Config.RESULTS_DIR(), f"detailed_results_{args.dataset}_{args.metric}_{timestamp}.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"Detailed results saved to CSV: {detailed_csv_path}")
    print(f"Inference completed. Results saved with timestamp: {timestamp}")
    print(f"Dataset used: {args.dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with CircScale model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to heuristic vocabulary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--inductive", action="store_true", help="Use inductive setup")
    parser.add_argument("--test_designs", type=str, default="test_designs.txt",
                        help="File with test design names for inductive setup")
    parser.add_argument("--metric", type=str, default="Delay", choices=["Delay", "Area"], help="QoR metric name")
    parser.add_argument("--plot_trajectories", action="store_true", help="Plot QoR trajectories")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device index to use")
    parser.add_argument("--dataset", type=str, choices=['original', 'epfl', 'openabc', 'iscas85', 'iscas89', 'opencores'],
                        default='original', help="Dataset to use for inference")

    parser.add_argument("--use_multiscale", action="store_true", help="Use multi-scale feature fusion")
    parser.add_argument("--max_hop", type=int, default=3, help="Maximum hop for multi-scale fusion")
    parser.add_argument("--use_dynamic_interaction", action="store_true",
                        help="Use dynamic graph-sequence adaptive interaction")

    args = parser.parse_args()

    # Set CUDA device
    if torch.cuda.is_available():
        Config.CUDA_DEVICE = args.cuda_device

    inference(args)