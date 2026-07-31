import os
import sys
import time
import torch

sys.path.append('/Users/jieun/Desktop/GIT/Research')

from ADD_NeuralODE_Open_neuralode_3_3 import (
    RobustGradientSpikeDetector,
    DATASET_CONFIG,
    unified_pipeline,
    plot_detection_summary,
    plot_top_example_from_metrics
)

def main():
    print('=' * 70)
    print('🚀 Running Full Neural ODE Anomaly Detection Simulation')
    print('=' * 70)
    
    dataset_name = 'SWaT'
    cfg = DATASET_CONFIG.get(dataset_name, {
        'latent_dim': 64,
        'hidden_dim': 256,
        'time_emb_dim': 12,
        'beta': 0.01,
        'batch_size': 256,
        'window_size': 120,
        'num_epochs': 50
    })
    
    dataset_path = './SWaT.pk'
    if not os.path.exists(dataset_path):
        alt_path = '/Users/jieun/Desktop/GIT/Research/CSV data/GT/GT_final.pk'
        if os.path.exists(alt_path):
            dataset_path = alt_path
            
    print(f"📊 Dataset Path: {dataset_path}")
    print(f"⚙️ Config: batch_size={cfg['batch_size']}, window_size={cfg['window_size']}, num_epochs={cfg['num_epochs']}")
    
    model, history, metrics, data_dict = unified_pipeline(
        dataset_path=dataset_path,
        model_class=RobustGradientSpikeDetector,
        model_kwargs={
            'latent_dim': cfg['latent_dim'],
            'hidden_dim': cfg['hidden_dim'],
            'time_emb_dim': cfg['time_emb_dim'],
            'beta': cfg['beta'],
            'noise_enabled': True,
            'ode_rtol': 1e-3,
            'ode_atol': 1e-4
        },
        batch_size=cfg['batch_size'],
        window_size=cfg['window_size'],
        stride=10,             # 10x fast execution
        num_epochs=cfg['num_epochs'],
        val_interval=10,
        val_max_sections=100,
        auto_visualize=True,  # Automatically save detection graphs
        figure_dir="results/figs"
    )
    
    print('=' * 70)
    print('✅ Full Simulation Complete!')
    if metrics:
        print(f"📁 Metrics Artifact: {metrics.get('artifact_path')}")
        print(f"📊 Detection Summary Graph: {metrics.get('figure_path')}")
    print('=' * 70)

if __name__ == '__main__':
    main()
