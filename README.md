# SudokuRL: Solving Sudoku with GNN + PPO

본 저장소는 **Graph Neural Network (GNN)** 과 **Proximal Policy Optimization (PPO)** 를 이용해  
4x4 / 9x9 스도쿠 퍼즐을 강화학습으로 푸는 프로젝트입니다.

스도쿠 보드를 그래프로 인코딩하고, PyTorch Geometric 기반 GCN 정책 네트워크로  
유효한 숫자 채우기 행동을 학습합니다.

---

## 1. 프로젝트 개요

이 프로젝트의 목표는 다음과 같습니다.

- 스도쿠 퍼즐을 **강화학습 환경**으로 정의
- 각 칸/행/열/블록을 노드로 하는 **그래프 표현** 설계
- GCN 기반 **정책/가치 함수**를 학습하는 PPO 에이전트 구현
- 4x4 / 9x9 스도쿠에 대해 학습 및 데모 수행
- 학습 로그 및 그래프(리턴, 완성률)를 자동 저장

퍼즐 데이터는 외부 파일 없이, 코드에서 **무작위로 스도쿠 정답 보드를 생성하고  
힌트를 부분적으로 가린 퍼즐**을 만들어 사용합니다.

---

## 2. 주요 특징

- ✅ 4x4, 9x9 스도쿠 모두 지원 (`--size 4` 또는 `--size 9`)
- ✅ 스도쿠를 **그래프(노드/엣지)** 로 표현하고 GNN으로 처리
- ✅ PPO(클리핑, GAE, entropy 보너스)를 이용한 정책 학습
- ✅ Apple Silicon(MPS), CUDA, CPU 자동 선택
- ✅ JSON `config` 파일로 하이퍼파라미터 제어
- ✅ 학습 로그(`.npy`) 및 학습 곡선(`.png`) 자동 저장
- ✅ 학습된 정책으로 **데모 에피소드** 실행 (탐욕 정책)

---

3. 환경 및 요구사항

이 프로젝트는 아래 환경에서 테스트되었습니다.

3.1 Python & 라이브러리 버전

필수 라이브러리:

Python 3.9+

PyTorch

PyTorch Geometric (PyG)

NumPy

Matplotlib

설치 예시(CUDA/MPS 환경에 따라 버전은 조정 필요):

pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy matplotlib


⚠️ PyTorch Geometric 설치는 PyTorch 버전에 맞게 해야 합니다.
정확한 설치 명령은 https://pytorch-geometric.readthedocs.io
 에서 확인하세요.

4. 실행 방법
4.1 기본 실행 (config 없이)
4×4 스도쿠 학습
python sudoku_rl.py --size 4

9×9 스도쿠 학습
python sudoku_rl.py --size 9

4.2 실행 옵션 설명
옵션	설명
--size {4,9}	스도쿠 크기 선택 (4×4 또는 9×9)
--single_base	하나의 base solution만 사용하고, 퍼즐(힌트 위치)만 다양화
--single_puzzle	힌트가 주어지는 위치 패턴을 하나로 고정 (base solution이 바뀌어도 힌트 위치 동일)
--num_instances	학습에 사용할 (퍼즐, 정답) 쌍 개수
--config	JSON 형식의 config 파일 경로

이 옵션들은 CLI 인자가 우선이며, config 파일에도 동일 키가 있을 경우 CLI 값이 적용됩니다.

4.3 config 파일을 사용하는 실행 예시

예) 4×4 스도쿠를 config 기반으로 실행:

python sudoku_rl.py --size 4 --config configs/sudoku4.json

4.4 4×4 config 파일 예시 (configs/sudoku4.json)
{
  "seed": 0,
  "size": 4,
  "num_instances": 300,
  "single_base": true,
  "single_puzzle": true,
  "log_dir": "./logs_4x4",

  "gamma": 0.99,
  "clip_eps": 0.2,
  "lr": 0.0005,
  "value_coef": 0.5,
  "entropy_coef": 0.001,

  "max_steps_4x4": 20,
  "in_dim_4x4": 12,
  "hidden_dim_4x4": 64,
  "num_layers_4x4": 3,
  "rollout_steps_4x4": 128,
  "ppo_epochs_4x4": 4,
  "batch_size_4x4": 16,
  "num_updates_4x4": 300
}

✔️ 설명

single_base: true 면 정답 solution은 고정

single_puzzle: true 면 힌트 위치 패턴도 고정

즉, 하나의 퍼즐 구조를 반복 학습하는 설정 → 단일 퍼즐 학습에 최적화됨

lr, entropy_coef, rollout_steps 등 PPO 핵심 파라미터는 config로 변경 가능
