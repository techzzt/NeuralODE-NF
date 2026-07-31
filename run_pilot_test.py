import os
import sys
import time
import torch

sys.path.append('/Users/jieun/Desktop/GIT/Research')

from ADD_NeuralODE_Open_neuralode_3_3 import (
    RobustGradientSpikeDetector,
    unified_load_and_prepare_data,
    unified_train,
    unified_evaluate,
    save_obj
)

def main():
    print('=' * 70)
    print('🚀 CODAR Pilot Test (Fast Verification Run)')
    print('=' * 70)
    start_time = time.time()
    
    dataset_path = '/Users/jieun/Desktop/GIT/Research/CSV data/GT/GT_final.pk'
    if not os.path.exists(dataset_path):
        dataset_path = 'SWaT.pk'
        
    print('[1/4] Loading dataset: ' + str(dataset_path))
    data_dict, data_type, input_dim = unified_load_and_prepare_data(
        dataset_path=dataset_path,
        batch_size=128,
        window_size=30
    )
    if data_dict is None or input_dim is None:
        print('❌ Dataset loading failed!')
        return
        
    print('  ✓ Data loaded successfully!')
    print('  ✓ Input dimension: ' + str(input_dim))
    
    print('[2/4] Initializing RobustGradientSpikeDetector Model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGradientSpikeDetector(
        input_dim=input_dim,
        latent_dim=24,
        hidden_dim=48,
        time_emb_dim=8,
        beta=0.01,
        noise_enabled=True
    ).to(device)
    
    print('[3/4] Running Fast Pilot Training (3 Epochs)...')
    model, history = unified_train(
        model=model,
        data_dict=data_dict,
        num_epochs=3,
        device=device,
        val_interval=None
    )
    
    print('[4/4] Evaluating and Saving Results...')
    metrics = unified_evaluate(model, data_dict, device=device)
    results_dir = '/Users/jieun/Desktop/GIT/Research/results'
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, 'PilotTest_CODAR.pkl')
    save_obj(metrics, save_path)
    
    elapsed = time.time() - start_time
    print('=' * 70)
    print('✅ Pilot test completed successfully in ' + str(round(elapsed, 2)) + ' seconds!')
    print('📁 Result artifact saved to: ' + str(save_path))
    print('=' * 70)

if __name__ == '__main__':
    main()
