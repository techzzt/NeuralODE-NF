import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTM_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Decoder
        self.decoder_fc = nn.Linear(hidden_dim // 2, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # Encoder
        encoded, _ = self.encoder(x)
        encoded = self.encoder_fc(encoded)
        
        # Decoder
        decoded = self.decoder_fc(encoded)
        decoded, _ = self.decoder(decoded)
        
        return decoded

class BaselineModels:
    def __init__(self):
        self.models = {
            'OneClassSVM': OneClassSVM(nu=0.1, kernel='rbf', gamma='scale'),
            'IsolationForest': IsolationForest(contamination=0.1, random_state=42),
            'LSTM_Autoencoder': None  # Will be initialized per dataset
        }
        self.scalers = {}
        self.results = {}
    
    def prepare_data(self, data, labels, test_size=0.3, random_state=42):
        """데이터 전처리 및 분할"""
        # 데이터 정규화
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            data_scaled, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_lstm_autoencoder(self, X_train, X_test, input_dim, sequence_length=10):
        """LSTM Autoencoder 훈련"""
        # 시퀀스 데이터 생성
        def create_sequences(data, seq_len):
            sequences = []
            for i in range(len(data) - seq_len + 1):
                sequences.append(data[i:i+seq_len])
            return np.array(sequences)
        
        X_train_seq = create_sequences(X_train, sequence_length)
        X_test_seq = create_sequences(X_test, sequence_length)
        
        # 모델 초기화
        model = LSTM_Autoencoder(input_dim, hidden_dim=64, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 훈련
        X_train_tensor = torch.FloatTensor(X_train_seq)
        X_test_tensor = torch.FloatTensor(X_test_seq)
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            reconstructed = model(X_train_tensor)
            loss = criterion(reconstructed, X_train_tensor)
            loss.backward()
            optimizer.step()
        
        # 이상 점수 계산 (재구성 오차)
        model.eval()
        with torch.no_grad():
            train_reconstructed = model(X_train_tensor)
            test_reconstructed = model(X_test_tensor)
            
            train_scores = torch.mean((X_train_tensor - train_reconstructed) ** 2, dim=(1, 2)).numpy()
            test_scores = torch.mean((X_test_tensor - test_reconstructed) ** 2, dim=(1, 2)).numpy()
        
        return test_scores, train_scores
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """모델 성능 평가"""
        try:
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            accuracy = accuracy_score(y_true, y_pred)
            
            # AUC 계산 (이상 점수가 있는 경우)
            if hasattr(self, 'anomaly_scores') and len(self.anomaly_scores) > 0:
                try:
                    auc = roc_auc_score(y_true, self.anomaly_scores)
                except:
                    auc = 0.0
            else:
                auc = 0.0
            
            return {
                'F1-Score': f1,
                'Precision': precision,
                'Recall': recall,
                'Accuracy': accuracy,
                'AUC-ROC': auc
            }
        except Exception as e:
            print(f"평가 오류 ({model_name}): {e}")
            return {
                'F1-Score': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'Accuracy': 0.0,
                'AUC-ROC': 0.0
            }
    
    def run_baseline_models(self, dataset_name, X_train, X_test, y_train, y_test):
        """기본 모델들 실행"""
        print(f"\n=== {dataset_name} 데이터셋 Baseline 모델 성능 ===")
        
        results = {}
        
        # 1. One-Class SVM
        try:
            print("One-Class SVM 훈련 중...")
            self.models['OneClassSVM'].fit(X_train)
            y_pred_svm = self.models['OneClassSVM'].predict(X_test)
            y_pred_svm = (y_pred_svm == -1).astype(int)  # -1을 이상으로 변환
            
            results['OneClassSVM'] = self.evaluate_model(y_test, y_pred_svm, 'OneClassSVM')
            print(f"One-Class SVM F1: {results['OneClassSVM']['F1-Score']:.4f}")
        except Exception as e:
            print(f"One-Class SVM 오류: {e}")
            results['OneClassSVM'] = {'F1-Score': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'Accuracy': 0.0, 'AUC-ROC': 0.0}
        
        # 2. Isolation Forest
        try:
            print("Isolation Forest 훈련 중...")
            self.models['IsolationForest'].fit(X_train)
            y_pred_if = self.models['IsolationForest'].predict(X_test)
            y_pred_if = (y_pred_if == -1).astype(int)  # -1을 이상으로 변환
            
            results['IsolationForest'] = self.evaluate_model(y_test, y_pred_if, 'IsolationForest')
            print(f"Isolation Forest F1: {results['IsolationForest']['F1-Score']:.4f}")
        except Exception as e:
            print(f"Isolation Forest 오류: {e}")
            results['IsolationForest'] = {'F1-Score': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'Accuracy': 0.0, 'AUC-ROC': 0.0}
        
        # 3. LSTM Autoencoder
        try:
            print("LSTM Autoencoder 훈련 중...")
            input_dim = X_train.shape[1]
            anomaly_scores, _ = self.train_lstm_autoencoder(X_train, X_test, input_dim)
            
            # 이상 점수를 기반으로 예측
            threshold = np.percentile(anomaly_scores, 90)  # 상위 10%를 이상으로
            y_pred_lstm = (anomaly_scores > threshold).astype(int)
            
            # 길이 맞추기
            min_len = min(len(y_test), len(y_pred_lstm))
            y_test_adj = y_test[:min_len]
            y_pred_lstm_adj = y_pred_lstm[:min_len]
            
            results['LSTM_Autoencoder'] = self.evaluate_model(y_test_adj, y_pred_lstm_adj, 'LSTM_Autoencoder')
            print(f"LSTM Autoencoder F1: {results['LSTM_Autoencoder']['F1-Score']:.4f}")
        except Exception as e:
            print(f"LSTM Autoencoder 오류: {e}")
            results['LSTM_Autoencoder'] = {'F1-Score': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'Accuracy': 0.0, 'AUC-ROC': 0.0}
        
        return results

def load_smd_data():
    """SMD 데이터 로드"""
    try:
        with open('./datasets/SMD/ServerMachineDataset/SMD.pk', 'rb') as f:
            data = pickle.load(f)
        
        # SMD 데이터 구조: x_trn, x_tst, lab_tst (리스트 형태)
        if isinstance(data, dict) and 'x_trn' in data and 'x_tst' in data:
            # 모든 섹션을 하나로 합치기
            x_train_list = data['x_trn']  # 28개 섹션
            x_test_list = data['x_tst']   # 28개 섹션
            y_test_list = data['lab_tst'] # 28개 섹션
            
            # 전체 데이터 합치기
            x_train = np.vstack(x_train_list)  # 모든 섹션 합치기
            x_test = np.vstack(x_test_list)    # 모든 섹션 합치기
            y_test = np.hstack(y_test_list)    # 모든 섹션 합치기
            
            # 훈련 라벨은 정상 데이터로 가정
            y_train = np.zeros(len(x_train))
            
            print(f"SMD 전체 데이터: 훈련 {x_train.shape}, 테스트 {x_test.shape}")
            
            return x_train, x_test, y_train, y_test
        else:
            print("SMD 데이터 구조를 파악할 수 없습니다.")
            return None, None, None, None
        
    except Exception as e:
        print(f"SMD 데이터 로드 오류: {e}")
        return None, None, None, None

def load_trimSyn_data():
    """trimSyn 데이터 로드"""
    try:
        with open('./datasets/trimSyn/trimSyn.pk', 'rb') as f:
            data = pickle.load(f)
        
        # trimSyn 데이터 구조: x_trn, x_tst, lab_tst (리스트 형태)
        if isinstance(data, dict) and 'x_trn' in data and 'x_tst' in data:
            # 모든 섹션을 하나로 합치기
            x_train_list = data['x_trn']  # 5개 섹션
            x_test_list = data['x_tst']   # 5개 섹션
            y_test_list = data['lab_tst'] # 5개 섹션
            
            # 전체 데이터 합치기
            x_train = np.vstack(x_train_list)  # 모든 섹션 합치기
            x_test = np.vstack(x_test_list)    # 모든 섹션 합치기
            y_test = np.hstack(y_test_list)    # 모든 섹션 합치기
            
            # 훈련 라벨은 정상 데이터로 가정
            y_train = np.zeros(len(x_train))
            
            print(f"trimSyn 전체 데이터: 훈련 {x_train.shape}, 테스트 {x_test.shape}")
            
            return x_train, x_test, y_train, y_test
        else:
            print("trimSyn 데이터 구조를 파악할 수 없습니다.")
            return None, None, None, None
        
    except Exception as e:
        print(f"trimSyn 데이터 로드 오류: {e}")
        return None, None, None, None

def load_wadi_data():
    """WADI 데이터 로드"""
    try:
        with open('./datasets/WADI/WADI.pk', 'rb') as f:
            data = pickle.load(f)
        
        # WADI 데이터 구조: x_trn, x_tst, lab_tst (numpy 배열)
        if isinstance(data, dict) and 'x_trn' in data and 'x_tst' in data:
            x_train_full = data['x_trn']  # (1209601, 123)
            x_test_full = data['x_tst']   # (172801, 123)
            y_test_full = data['lab_tst'] # (172801,)
            
            # WADI는 너무 크므로 적절히 샘플링 (하지만 더 많은 데이터 사용)
            train_sample_size = min(200000, len(x_train_full))  # 더 많은 훈련 데이터
            test_sample_size = min(50000, len(x_test_full))     # 더 많은 테스트 데이터
            
            # 랜덤 샘플링
            np.random.seed(42)
            train_indices = np.random.choice(len(x_train_full), train_sample_size, replace=False)
            test_indices = np.random.choice(len(x_test_full), test_sample_size, replace=False)
            
            x_train = x_train_full[train_indices]
            x_test = x_test_full[test_indices]
            y_test = y_test_full[test_indices]
            
            # 훈련 라벨은 정상 데이터로 가정
            y_train = np.zeros(len(x_train))
            
            print(f"WADI 데이터 샘플링: 훈련 {len(x_train)}, 테스트 {len(x_test)}")
            
            return x_train, x_test, y_train, y_test
        else:
            print("WADI 데이터 구조를 파악할 수 없습니다.")
            return None, None, None, None
        
    except Exception as e:
        print(f"WADI 데이터 로드 오류: {e}")
        return None, None, None, None

# MSL과 SMAP 데이터는 현재 사용 불가능하므로 제거

def main():
    """메인 실행 함수"""
    print("=== Baseline 모델 성능 비교 ===")
    
    # 데이터셋 로드 함수들 (SMD만 테스트)
    datasets = {
        'SMD': load_smd_data
    }
    
    # Baseline 모델 초기화
    baseline = BaselineModels()
    
    # 전체 결과 저장
    all_results = {}
    
    for dataset_name, load_func in datasets.items():
        print(f"\n{'='*50}")
        print(f"{dataset_name} 데이터셋 처리 중...")
        
        # 데이터 로드
        x_train, x_test, y_train, y_test = load_func()
        
        if x_train is None:
            print(f"❌ {dataset_name} 데이터 로드 실패")
            continue
        
        print(f"✅ {dataset_name} 데이터 로드 성공")
        print(f"   훈련 데이터: {x_train.shape}")
        print(f"   테스트 데이터: {x_test.shape}")
        print(f"   이상 비율: {np.mean(y_test):.3f}")
        
        # 데이터 전처리
        X_train, X_test, y_train_adj, y_test_adj, scaler = baseline.prepare_data(
            x_test, y_test, test_size=0.3
        )
        
        # Baseline 모델 실행
        results = baseline.run_baseline_models(
            dataset_name, X_train, X_test, y_train_adj, y_test_adj
        )
        
        all_results[dataset_name] = results
        
        # 결과 출력
        print(f"\n{dataset_name} 결과:")
        for model_name, metrics in results.items():
            print(f"  {model_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    # 전체 결과 요약
    print(f"\n{'='*50}")
    print("전체 결과 요약")
    print(f"{'='*50}")
    
    # 결과를 DataFrame으로 변환
    summary_data = []
    for dataset_name, results in all_results.items():
        for model_name, metrics in results.items():
            row = {'Dataset': dataset_name, 'Model': model_name}
            row.update(metrics)
            summary_data.append(row)
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # F1-Score 기준으로 정렬
        df_summary = df_summary.sort_values(['Dataset', 'F1-Score'], ascending=[True, False])
        
        print("\nF1-Score 기준 정렬:")
        print(df_summary[['Dataset', 'Model', 'F1-Score', 'Precision', 'Recall', 'Accuracy']].to_string(index=False))
        
        # 각 데이터셋별 최고 성능 모델
        print("\n각 데이터셋별 최고 F1-Score 모델:")
        best_models = df_summary.loc[df_summary.groupby('Dataset')['F1-Score'].idxmax()]
        print(best_models[['Dataset', 'Model', 'F1-Score']].to_string(index=False))
        
        # 결과를 CSV로 저장
        df_summary.to_csv('baseline_results.csv', index=False)
        print(f"\n결과가 'baseline_results.csv'에 저장되었습니다.")
    else:
        print("❌ 결과가 없습니다.")

if __name__ == "__main__":
    main()
