import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
from dataclasses import dataclass
from typing import List, Tuple

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

import matplotlib.pyplot as plt


def get_device():
    if torch.backends.mps.is_available():
        print("[Device] Using Apple MPS (Metal)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[Device] Using CUDA")
        return torch.device("cuda")
    else:
        print("[Device] Using CPU")
        return torch.device("cpu")


DEVICE = get_device()


def load_config(path: str):
    if path is None:
        return {}
    if not os.path.exists(path):
        print(f"[Config] 파일을 찾을 수 없습니다: {path} (기본값 사용)")
        return {}
    with open(path, "r") as f:
        cfg = json.load(f)
    print(f"[Config] Loaded from {path}")
    return cfg


@dataclass
class SudokuInstance:
    puzzle: np.ndarray
    solution: np.ndarray


def violates_rule_4x4(board: np.ndarray, r: int, c: int, d: int) -> bool:
    if d in board[r, :]:
        return True
    if d in board[:, c]:
        return True
    br = (r // 2) * 2
    bc = (c // 2) * 2
    if d in board[br:br+2, bc:bc+2]:
        return True
    return False


def generate_puzzle_from_solution_4x4(solution: np.ndarray,
                                      num_givens: int) -> np.ndarray:
    puzzle = np.zeros_like(solution)
    all_indices = np.arange(16)
    np.random.shuffle(all_indices)
    chosen = all_indices[:num_givens]
    for idx in chosen:
        r = idx // 4
        c = idx % 4
        puzzle[r, c] = solution[r, c]
    return puzzle


def randomize_solution_from_base_4x4(base_solution: np.ndarray) -> np.ndarray:
    """
    4x4 base solution에 대해:
    - 숫자 permute(1~4)
    - 행 band(0-1,2-3) permute + band 내 row permute
    - 열 band(0-1,2-3) permute + band 내 col permute
    """
    s = base_solution.copy()

    digits = np.arange(1, 5)
    perm_digits = np.random.permutation(digits)
    s2 = np.zeros_like(s)
    for old_d, new_d in zip(digits, perm_digits):
        s2[s == old_d] = new_d
    s = s2

    band_perm = np.random.permutation([0, 1])
    row_order = []
    for b in band_perm:
        base_rows = [2 * b + 0, 2 * b + 1]
        perm_within = np.random.permutation(base_rows)
        row_order.extend(perm_within)
    s = s[row_order, :]

    stack_perm = np.random.permutation([0, 1])
    col_order = []
    for s_idx in stack_perm:
        base_cols = [2 * s_idx + 0, 2 * s_idx + 1]
        perm_within = np.random.permutation(base_cols)
        col_order.extend(perm_within)
    s = s[:, col_order]

    return s


def get_example_instances_4x4(num_instances: int = 100,
                              single_base: bool = False,
                              single_puzzle: bool = False) -> List[SudokuInstance]:
    base_solution = np.array([
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ], dtype=int)

    instances: List[SudokuInstance] = []

    fixed_chosen = None

    if single_puzzle:
        fixed_num_givens = 8
        all_indices = np.arange(16)
        np.random.shuffle(all_indices)
        fixed_chosen = all_indices[:fixed_num_givens]

    for _ in range(num_instances):
        if single_base:
            solution = base_solution.copy()
        else:
            solution = randomize_solution_from_base_4x4(base_solution)

        if single_puzzle:
            puzzle = np.zeros_like(solution)
            for idx in fixed_chosen:
                r = idx // 4
                c = idx % 4
                puzzle[r, c] = solution[r, c]
        else:
            num_givens = 8
            puzzle = generate_puzzle_from_solution_4x4(solution, num_givens)

        instances.append(SudokuInstance(puzzle=puzzle, solution=solution))

    return instances


def violates_rule_9x9(board: np.ndarray, r: int, c: int, d: int) -> bool:
    if d in board[r, :]:
        return True
    if d in board[:, c]:
        return True
    br = (r // 3) * 3
    bc = (c // 3) * 3
    if d in board[br:br+3, bc:bc+3]:
        return True
    return False


def generate_puzzle_from_solution_9x9(solution: np.ndarray,
                                      num_givens: int) -> np.ndarray:
    puzzle = np.zeros_like(solution)
    all_indices = np.arange(81)
    np.random.shuffle(all_indices)
    chosen = all_indices[:num_givens]
    for idx in chosen:
        r = idx // 9
        c = idx % 9
        puzzle[r, c] = solution[r, c]
    return puzzle


def randomize_solution_from_base_9x9(base_solution: np.ndarray) -> np.ndarray:
    s = base_solution.copy()

    digits = np.arange(1, 10)
    perm_digits = np.random.permutation(digits)
    s2 = np.zeros_like(s)
    for old_d, new_d in zip(digits, perm_digits):
        s2[s == old_d] = new_d
    s = s2

    band_perm = np.random.permutation([0, 1, 2])
    row_order = []
    for b in band_perm:
        base_rows = [3 * b + 0, 3 * b + 1, 3 * b + 2]
        perm_within = np.random.permutation(base_rows)
        row_order.extend(perm_within)
    s = s[row_order, :]

    stack_perm = np.random.permutation([0, 1, 2])
    col_order = []
    for s_idx in stack_perm:
        base_cols = [3 * s_idx + 0, 3 * s_idx + 1, 3 * s_idx + 2]
        perm_within = np.random.permutation(base_cols)
        col_order.extend(perm_within)
    s = s[:, col_order]

    return s


def get_example_instances_9x9(num_instances: int = 200,
                              single_base: bool = False,
                              single_puzzle: bool = False) -> List[SudokuInstance]:
    base_solution = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ], dtype=int)

    instances: List[SudokuInstance] = []

    fixed_chosen = None

    if single_puzzle:
        fixed_num_givens = 60
        all_indices = np.arange(81)
        np.random.shuffle(all_indices)
        fixed_chosen = all_indices[:fixed_num_givens]

    for _ in range(num_instances):
        if single_base:
            solution = base_solution.copy()
        else:
            solution = randomize_solution_from_base_9x9(base_solution)

        if single_puzzle:
            puzzle = np.zeros_like(solution)
            for idx in fixed_chosen:
                r = idx // 9
                c = idx % 9
                puzzle[r, c] = solution[r, c]
        else:
            num_givens = 60
            puzzle = generate_puzzle_from_solution_9x9(solution, num_givens)

        instances.append(SudokuInstance(puzzle=puzzle, solution=solution))

    return instances


class Sudoku4Env:
    def __init__(self, instances: List[SudokuInstance], max_steps: int = 20):
        self.instances = instances
        self.max_steps = max_steps

        self.current: SudokuInstance = None
        self.state: np.ndarray = None
        self.given_mask: np.ndarray = None
        self.step_count: int = 0

    def _is_complete(self) -> bool:
        return (self.state != 0).all()

    def _is_correct(self) -> bool:
        return np.array_equal(self.state, self.current.solution)

    def compute_valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(16 * 4, dtype=bool)
        for cell_idx in range(16):
            r = cell_idx // 4
            c = cell_idx % 4
            if self.given_mask[r, c] or self.state[r, c] != 0:
                continue
            for d in range(1, 5):
                if not violates_rule_4x4(self.state, r, c, d):
                    a = cell_idx * 4 + (d - 1)
                    mask[a] = True
        return mask

    def _state_to_graph(self) -> Data:
        """
        4x4:
        - cell node: 16
        - row meta: 4
        - col meta: 4
        - block meta(2x2): 4
        => 총 28 nodes
        feature dim (12):
          [0:5): digit one-hot (0~4)
          [5]:   is_given
          [6]:   is_meta
          [7:10): meta_type (row/col/block) one-hot
          [10]:  row_norm
          [11]:  col_norm
        """
        board = self.state
        puzzle = self.current.puzzle

        num_cells = 16
        num_row_meta = 4
        num_col_meta = 4
        num_blk_meta = 4
        num_meta = num_row_meta + num_col_meta + num_blk_meta
        num_nodes = num_cells + num_meta
        feat_dim = 12

        x = np.zeros((num_nodes, feat_dim), dtype=np.float32)

        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                digit = board[r, c]
                x[idx, digit] = 1.0
                x[idx, 5] = 1.0 if puzzle[r, c] != 0 else 0.0
                x[idx, 6] = 0.0
                x[idx, 10] = r / 3.0
                x[idx, 11] = c / 3.0

        base = num_cells
        for r in range(4):
            node_idx = base + r
            used = board[r, :]
            for d in used:
                if d != 0:
                    x[node_idx, d] = 1.0
            x[node_idx, 6] = 1.0
            x[node_idx, 7] = 1.0
            x[node_idx, 10] = r / 3.0

        base = num_cells + num_row_meta
        for c in range(4):
            node_idx = base + c
            used = board[:, c]
            for d in used:
                if d != 0:
                    x[node_idx, d] = 1.0
            x[node_idx, 6] = 1.0
            x[node_idx, 8] = 1.0

        base = num_cells + num_row_meta + num_col_meta
        blk_id = 0
        for br in range(2):
            for bc in range(2):
                node_idx = base + blk_id
                used = board[2*br:2*br+2, 2*bc:2*bc+2].reshape(-1)
                for d in used:
                    if d != 0:
                        x[node_idx, d] = 1.0
                x[node_idx, 6] = 1.0
                x[node_idx, 9] = 1.0
                x[node_idx, 10] = br / 1.0
                x[node_idx, 11] = bc / 1.0
                blk_id += 1

        edges_src, edges_dst = [], []

        def add_edge(u, v):
            edges_src.append(u)
            edges_dst.append(v)

        row_base = num_cells
        for r in range(4):
            row_node = row_base + r
            for c in range(4):
                cell = r * 4 + c
                add_edge(cell, row_node)
                add_edge(row_node, cell)

        col_base = num_cells + num_row_meta
        for c in range(4):
            col_node = col_base + c
            for r in range(4):
                cell = r * 4 + c
                add_edge(cell, col_node)
                add_edge(col_node, cell)

        blk_base = num_cells + num_row_meta + num_col_meta
        blk_id = 0
        for br in range(2):
            for bc in range(2):
                blk_node = blk_base + blk_id
                for rr in range(2):
                    for cc in range(2):
                        r = 2*br + rr
                        c = 2*bc + cc
                        cell = r * 4 + c
                        add_edge(cell, blk_node)
                        add_edge(blk_node, cell)
                blk_id += 1

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        cell_mask = np.zeros(num_nodes, dtype=bool)
        cell_mask[:num_cells] = True

        valid_mask = self.compute_valid_action_mask()

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
        )
        data.cell_mask = torch.from_numpy(cell_mask)
        data.valid_mask = torch.from_numpy(valid_mask)
        return data

    def render(self):
        board = self.state
        puzzle = self.current.puzzle
        print("-" * 17)
        for r in range(4):
            row_str = ""
            for c in range(4):
                v = board[r, c]
                if v == 0:
                    cell = "."
                else:
                    if puzzle[r, c] != 0:
                        cell = f"({v})"
                    else:
                        cell = f" {v} "
                row_str += cell + " "
                if c == 1:
                    row_str += "| "
            print(row_str)
            if r == 1:
                print("-" * 17)
        print("-" * 17)

    def reset(self) -> Data:
        self.current = random.choice(self.instances)
        self.state = self.current.puzzle.copy()
        self.given_mask = self.current.puzzle != 0
        self.step_count = 0
        return self._state_to_graph()

    def step(self, action: int) -> Tuple[Data, float, bool, dict]:
        valid_mask = self.compute_valid_action_mask()
        if not valid_mask.any():
            reward = -2.0
            done = True
            quality = int((self.state == self.current.solution).sum())
            info = {
                "quality": quality,
                "reason": "no_valid_action",
                "is_complete": self._is_complete()
            }
            return self._state_to_graph(), reward, done, info

        self.step_count += 1
        a = int(action)
        assert 0 <= a < 16 * 4

        cell_idx = a // 4
        digit = (a % 4) + 1

        r = cell_idx // 4
        c = cell_idx % 4

        reward = 0.0
        done = False

        if self.given_mask[r, c]:
            reward = -1.0
            done = True
        elif self.state[r, c] != 0:
            reward = -0.5
        elif violates_rule_4x4(self.state, r, c, digit):
            reward = -1.0
            done = True
        else:
            self.state[r, c] = digit
            if digit == self.current.solution[r, c]:
                reward = +1.0
            else:
                reward = -0.2

        if not done:
            if self._is_complete():
                reward += 10.0
                done = True
            elif self.step_count >= self.max_steps:
                done = True

        quality = int((self.state == self.current.solution).sum())
        info = {
            "quality": quality,
            "is_complete": self._is_complete()
        }
        return self._state_to_graph(), float(reward), done, info


class Sudoku9Env:
    def __init__(self, instances: List[SudokuInstance], max_steps: int = 200):
        self.instances = instances
        self.max_steps = max_steps

        self.current: SudokuInstance = None
        self.state: np.ndarray = None
        self.given_mask: np.ndarray = None
        self.step_count: int = 0

    def _is_complete(self) -> bool:
        return (self.state != 0).all()

    def _is_correct(self) -> bool:
        return np.array_equal(self.state, self.current.solution)

    def compute_valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(81 * 9, dtype=bool)
        for cell_idx in range(81):
            r = cell_idx // 9
            c = cell_idx % 9
            if self.given_mask[r, c] or self.state[r, c] != 0:
                continue
            for d in range(1, 10):
                if not violates_rule_9x9(self.state, r, c, d):
                    a = cell_idx * 9 + (d - 1)
                    mask[a] = True
        return mask

    def _state_to_graph(self) -> Data:
        """
        9x9:
        - cell: 81
        - row meta: 9
        - col meta: 9
        - block meta: 9
        => 108 nodes
        feat_dim=14:
          [0:10): digit one-hot (0=empty)
          [10]: is_given
          [11]: is_meta
          [12]: row_norm
          [13]: col_norm
        """
        board = self.state
        puzzle = self.current.puzzle

        num_cells = 81
        num_row_meta = 9
        num_col_meta = 9
        num_blk_meta = 9
        num_meta = num_row_meta + num_col_meta + num_blk_meta
        num_nodes = num_cells + num_meta
        feat_dim = 14

        x = np.zeros((num_nodes, feat_dim), dtype=np.float32)

        for r in range(9):
            for c in range(9):
                idx = r * 9 + c
                digit = board[r, c]
                if digit == 0:
                    x[idx, 0] = 1.0
                else:
                    x[idx, digit] = 1.0
                x[idx, 10] = 1.0 if puzzle[r, c] != 0 else 0.0
                x[idx, 11] = 0.0
                x[idx, 12] = r / 8.0
                x[idx, 13] = c / 8.0

        base = num_cells
        for r in range(9):
            node_idx = base + r
            used = board[r, :]
            for d in used:
                if d != 0:
                    x[node_idx, d] = 1.0
            x[node_idx, 11] = 1.0
            x[node_idx, 12] = r / 8.0

        base = num_cells + num_row_meta
        for c in range(9):
            node_idx = base + c
            used = board[:, c]
            for d in used:
                if d != 0:
                    x[node_idx, d] = 1.0
            x[node_idx, 11] = 1.0
            x[node_idx, 13] = c / 8.0

        base = num_cells + num_row_meta + num_col_meta
        blk_id = 0
        for br in range(3):
            for bc in range(3):
                node_idx = base + blk_id
                used = board[3*br:3*br+3, 3*bc:3*bc+3].reshape(-1)
                for d in used:
                    if d != 0:
                        x[node_idx, d] = 1.0
                x[node_idx, 11] = 1.0
                x[node_idx, 12] = br / 2.0
                x[node_idx, 13] = bc / 2.0
                blk_id += 1

        edges_src, edges_dst = [], []

        def add_edge(u, v):
            edges_src.append(u)
            edges_dst.append(v)

        row_base = num_cells
        for r in range(9):
            row_node = row_base + r
            for c in range(9):
                cell = r * 9 + c
                add_edge(cell, row_node)
                add_edge(row_node, cell)

        col_base = num_cells + num_row_meta
        for c in range(9):
            col_node = col_base + c
            for r in range(9):
                cell = r * 9 + c
                add_edge(cell, col_node)
                add_edge(col_node, cell)

        blk_base = num_cells + num_row_meta + num_col_meta
        blk_id = 0
        for br in range(3):
            for bc in range(3):
                blk_node = blk_base + blk_id
                for rr in range(3):
                    for cc in range(3):
                        r = 3*br + rr
                        c = 3*bc + cc
                        cell = r * 9 + c
                        add_edge(cell, blk_node)
                        add_edge(blk_node, cell)
                blk_id += 1

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        cell_mask = np.zeros(num_nodes, dtype=bool)
        cell_mask[:num_cells] = True

        valid_mask = self.compute_valid_action_mask()

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
        )
        data.cell_mask = torch.from_numpy(cell_mask)
        data.valid_mask = torch.from_numpy(valid_mask)
        return data

    def render(self):
        board = self.state
        puzzle = self.current.puzzle
        print("-" * 37)
        for r in range(9):
            row_str = ""
            for c in range(9):
                v = board[r, c]
                if v == 0:
                    cell = "."
                else:
                    if puzzle[r, c] != 0:
                        cell = f"({v})"
                    else:
                        cell = f" {v} "
                row_str += cell + " "
                if c in [2, 5]:
                    row_str += "| "
            print(row_str)
            if r in [2, 5]:
                print("-" * 37)
        print("-" * 37)

    def reset(self) -> Data:
        self.current = random.choice(self.instances)
        self.state = self.current.puzzle.copy()
        self.given_mask = self.current.puzzle != 0
        self.step_count = 0
        return self._state_to_graph()

    def step(self, action: int) -> Tuple[Data, float, bool, dict]:
        valid_mask = self.compute_valid_action_mask()
        if not valid_mask.any():
            reward = -2.0
            done = True
            quality = int((self.state == self.current.solution).sum())
            info = {
                "quality": quality,
                "reason": "no_valid_action",
                "is_complete": self._is_complete()
            }
            return self._state_to_graph(), reward, done, info

        self.step_count += 1
        a = int(action)
        assert 0 <= a < 81 * 9

        cell_idx = a // 9
        digit = (a % 9) + 1

        r = cell_idx // 9
        c = cell_idx % 9

        reward = 0.0
        done = False

        if self.given_mask[r, c]:
            reward = -1.0
            done = True
        elif self.state[r, c] != 0:
            reward = -0.5
        elif violates_rule_9x9(self.state, r, c, digit):
            reward = -1.0
            done = True
        else:
            self.state[r, c] = digit
            if digit == self.current.solution[r, c]:
                reward = +1.0
            else:
                reward = -0.2

        if not done:
            if self._is_complete():
                reward += 10.0
                done = True
            elif self.step_count >= self.max_steps:
                done = True

        quality = int((self.state == self.current.solution).sum())
        info = {
            "quality": quality,
            "is_complete": self._is_complete()
        }
        return self._state_to_graph(), float(reward), done, info


class SudokuPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_actions: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_feat = global_mean_pool(x, batch)

        logits = self.actor_head(graph_feat).squeeze(0)
        value = self.critic_head(graph_feat).squeeze()
        return logits, value


class PPOAgent:
    def __init__(self,
                 env,
                 policy: nn.Module,
                 board_cells: int,
                 device=DEVICE,
                 gamma=0.99,
                 clip_eps=0.2,
                 lr=3e-4,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 rollout_steps=256,
                 ppo_epochs=4,
                 batch_size=64):
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.board_cells = board_cells

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.episode_correct_history: List[int] = []
        self.episode_return_history: List[float] = []
        self.episode_complete_history: List[bool] = []
        self.total_episodes: int = 0

        self.update_avg_return: List[float] = []
        self.update_completion_rate: List[float] = []

    def collect_rollout(self):
        obs_list = []
        action_list = []
        logprob_list = []
        reward_list = []
        value_list = []
        done_list = []

        data = self.env.reset().to(self.device)

        episode_return = 0.0

        for _ in range(self.rollout_steps):
            with torch.no_grad():
                logits, value = self.policy(data)
                if hasattr(data, "valid_mask"):
                    mask = data.valid_mask.to(self.device).bool()
                    if mask.any():
                        logits = logits.masked_fill(~mask, -1e9)

                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            next_data, reward, done, info = self.env.step(action.item())
            next_data = next_data.to(self.device)

            obs_list.append(data)
            action_list.append(action)
            logprob_list.append(logprob)
            reward_list.append(reward)
            value_list.append(value)
            done_list.append(done)

            episode_return += reward

            if done:
                quality = int(info.get("quality", 0))
                is_complete = bool(info.get("is_complete", False))
                self.episode_correct_history.append(quality)
                self.episode_return_history.append(float(episode_return))
                self.episode_complete_history.append(is_complete)
                self.total_episodes += 1
                episode_return = 0.0

            data = next_data
            if done:
                data = self.env.reset().to(self.device)

        with torch.no_grad():
            logits, last_value = self.policy(data)
            if hasattr(data, "valid_mask"):
                mask = data.valid_mask.to(self.device).bool()
                if mask.any():
                    logits = logits.masked_fill(~mask, -1e9)

        rewards = torch.tensor(reward_list, dtype=torch.float32, device=self.device)
        values = torch.stack(value_list)
        dones = torch.tensor(done_list, dtype=torch.float32, device=self.device)
        logprobs = torch.stack(logprob_list)

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        gae = 0.0
        next_v = last_value
        for t in reversed(range(self.rollout_steps)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * mask - values[t]
            gae = delta + self.gamma * gae * mask
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_v = values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        buffer = {
            "obs": obs_list,
            "actions": torch.stack(action_list),
            "logprobs": logprobs,
            "returns": returns,
            "advantages": advantages,
        }
        return buffer

    def ppo_update(self, buffer):
        T = self.rollout_steps
        indices = np.arange(T)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]
                if len(mb_idx) == 0:
                    continue

                mb_obs = [buffer["obs"][i] for i in mb_idx]
                mb_actions = buffer["actions"][mb_idx].to(self.device)
                mb_old_logprobs = buffer["logprobs"][mb_idx].to(self.device)
                mb_returns = buffer["returns"][mb_idx].to(self.device)
                mb_advantages = buffer["advantages"][mb_idx].to(self.device)

                policy_loss_epoch = 0.0
                value_loss_epoch = 0.0
                entropy_epoch = 0.0
                count = 0

                for i in range(len(mb_idx)):
                    data = mb_obs[i].to(self.device)
                    action = mb_actions[i]
                    old_logprob = mb_old_logprobs[i]
                    ret = mb_returns[i]
                    adv = mb_advantages[i]

                    logits, value = self.policy(data)
                    if hasattr(data, "valid_mask"):
                        mask = data.valid_mask.to(self.device).bool()
                        if mask.any():
                            logits = logits.masked_fill(~mask, -1e9)

                    dist = Categorical(logits=logits)
                    logprob = dist.log_prob(action)
                    entropy = dist.entropy()

                    ratio = torch.exp(logprob - old_logprob)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                        1.0 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = F.mse_loss(value, ret)

                    policy_loss_epoch += policy_loss
                    value_loss_epoch += value_loss
                    entropy_epoch += entropy
                    count += 1

                policy_loss_epoch /= count
                value_loss_epoch /= count
                entropy_epoch /= count

                loss = (policy_loss_epoch
                        + self.value_coef * value_loss_epoch
                        - self.entropy_coef * entropy_epoch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

    def train(self, num_updates=500):
        for update in range(1, num_updates + 1):
            buffer = self.collect_rollout()
            self.ppo_update(buffer)

            if self.total_episodes >= 1:
                window = 10
                recent_returns = self.episode_return_history[-window:]
                recent_complete = self.episode_complete_history[-window:]

                avg_ret = float(np.mean(recent_returns))
                completion_rate = float(np.mean(recent_complete))

                self.update_avg_return.append(avg_ret)
                self.update_completion_rate.append(completion_rate)

            if update % 10 == 0:
                print(f"[Update {update}/{num_updates}] finished rollout & PPO update")

            if self.total_episodes >= 10 and self.total_episodes % 10 == 0:
                last10 = self.episode_correct_history[-10:]
                avg_correct = float(np.mean(last10))
                print(f"[Episodes {self.total_episodes-9}~{self.total_episodes}] "
                      f"평균 맞힌 칸 수: {avg_correct:.2f} / {self.board_cells}")


def demo_episode(agent: PPOAgent, env, max_steps=200, board_size=9):
    data = env.reset().to(agent.device)
    print("\n===== 데모 에피소드 시작 =====")
    print("[초기 퍼즐]")
    env.render()

    target_cells = 16 if board_size == 4 else 81

    for t in range(max_steps):
        with torch.no_grad():
            logits, value = agent.policy(data)
            if hasattr(data, "valid_mask"):
                mask = data.valid_mask.to(agent.device).bool()
                if mask.any():
                    logits = logits.masked_fill(~mask, -1e9)

            action = torch.argmax(logits)

        next_data, reward, done, info = env.step(int(action.item()))
        data = next_data.to(agent.device)

        print(f"\n--- step {t+1} ---")
        print(f"reward: {reward:.2f}, quality(정답과 같은 칸 수): {info['quality']}")
        env.render()

        if done:
            if info["quality"] == target_cells:
                print(f"✅ 퍼즐 완성! (step {t+1})")
            else:
                print(f"⛔ 종료 (step {t+1}, quality={info['quality']})")
            break
    else:
        print(f"⏱ max_steps({max_steps}) 도달")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=9, choices=[4, 9],
                        help="스도쿠 크기: 4 또는 9")
    parser.add_argument("--single_base", action="store_true",
                        help="True면 하나의 base solution에 대해 퍼즐만 다양화")
    parser.add_argument("--single_puzzle", action="store_true",
                        help="True면 가려지는 위치 패턴도 하나로 고정")
    parser.add_argument("--num_instances", type=int, default=300,
                        help="학습용 퍼즐/솔루션 쌍 개수")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 형식의 config 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    size = args.size
    if "size" in cfg:
        print(f"[Config] size (config={cfg['size']}, cli={args.size}) -> cli 우선 사용: {size}")

    gamma = cfg.get("gamma", 0.99)
    clip_eps = cfg.get("clip_eps", 0.2)
    lr = cfg.get("lr", 3e-4)
    value_coef = cfg.get("value_coef", 0.5)
    entropy_coef = cfg.get("entropy_coef", 0.01)

    num_instances = cfg.get("num_instances", args.num_instances)
    single_base = cfg.get("single_base", args.single_base)
    single_puzzle = cfg.get("single_puzzle", args.single_puzzle)

    log_dir = cfg.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    if size == 4:
        print("=== 4x4 Sudoku ===")

        max_steps = cfg.get("max_steps_4x4", 20)

        train_instances = get_example_instances_4x4(
            num_instances=num_instances,
            single_base=single_base,
            single_puzzle=single_puzzle,
        )
        env = Sudoku4Env(train_instances, max_steps=max_steps)

        in_dim = cfg.get("in_dim_4x4", 12)
        num_actions = 16 * 4
        hidden_dim = cfg.get("hidden_dim_4x4", 64)
        num_layers = cfg.get("num_layers_4x4", 3)

        rollout_steps = cfg.get("rollout_steps_4x4", 64)
        ppo_epochs = cfg.get("ppo_epochs_4x4", 4)
        batch_size = cfg.get("batch_size_4x4", 16)
        num_updates = cfg.get("num_updates_4x4", 1000)

        board_cells = 16

    else:
        print("=== 9x9 Sudoku ===")

        max_steps = cfg.get("max_steps_9x9", 200)

        train_instances = get_example_instances_9x9(
            num_instances=num_instances,
            single_base=single_base,
            single_puzzle=single_puzzle,
        )
        env = Sudoku9Env(train_instances, max_steps=max_steps)

        in_dim = cfg.get("in_dim_9x9", 14)
        num_actions = 81 * 9
        hidden_dim = cfg.get("hidden_dim_9x9", 256)
        num_layers = cfg.get("num_layers_9x9", 4)

        rollout_steps = cfg.get("rollout_steps_9x9", 256)
        ppo_epochs = cfg.get("ppo_epochs_9x9", 4)
        batch_size = cfg.get("batch_size_9x9", 64)
        num_updates = cfg.get("num_updates_9x9", 2000)

        board_cells = 81

    print("\n[Config Summary]")
    print(f"  size = {size}")
    print(f"  num_instances = {num_instances}, single_base = {single_base}, single_puzzle = {single_puzzle}")
    print(f"  max_steps = {max_steps}")
    print(f"  model: in_dim = {in_dim}, hidden_dim = {hidden_dim}, num_layers = {num_layers}")
    print(f"  PPO: gamma = {gamma}, clip_eps = {clip_eps}, lr = {lr}, "
          f"value_coef = {value_coef}, entropy_coef = {entropy_coef}")
    print(f"  rollout_steps = {rollout_steps}, ppo_epochs = {ppo_epochs}, "
          f"batch_size = {batch_size}, num_updates = {num_updates}")
    print(f"  log_dir = {log_dir}\n")

    policy = SudokuPolicy(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions
    )

    agent = PPOAgent(
        env,
        policy=policy,
        board_cells=board_cells,
        device=DEVICE,
        gamma=gamma,
        clip_eps=clip_eps,
        lr=lr,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        rollout_steps=rollout_steps,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
    )

    agent.train(num_updates=num_updates)

    np.save(os.path.join(log_dir, f"episode_return_size{size}.npy"),
            np.array(agent.episode_return_history, dtype=np.float32))
    np.save(os.path.join(log_dir, f"episode_complete_size{size}.npy"),
            np.array(agent.episode_complete_history, dtype=bool))
    np.save(os.path.join(log_dir, f"update_avg_return_size{size}.npy"),
            np.array(agent.update_avg_return, dtype=np.float32))
    np.save(os.path.join(log_dir, f"update_completion_rate_size{size}.npy"),
            np.array(agent.update_completion_rate, dtype=np.float32))

    updates = np.arange(1, len(agent.update_avg_return) + 1)

    plt.figure()
    plt.plot(updates, agent.update_avg_return)
    plt.xlabel("PPO Update")
    plt.ylabel("Avg Return (last 10 episodes)")
    plt.title(f"Return Curve (size={size})")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"return_curve_size{size}.png"))
    plt.close()

    plt.figure()
    plt.plot(updates, agent.update_completion_rate)
    plt.xlabel("PPO Update")
    plt.ylabel("Completion Rate (last 10 episodes)")
    plt.title(f"Board Completion Curve (size={size})")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"completion_curve_size{size}.png"))
    plt.close()

    print(f"[Log] Saved metrics and curves to: {log_dir}")

    if size == 4:
        base_solution = np.array([
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ], dtype=int)
        if single_base:
            test_solution = base_solution.copy()
        else:
            test_solution = randomize_solution_from_base_4x4(base_solution)
        num_givens = 8
        test_puzzle = generate_puzzle_from_solution_4x4(test_solution, num_givens)
        test_instance = SudokuInstance(puzzle=test_puzzle, solution=test_solution)
        test_env = Sudoku4Env([test_instance], max_steps=max_steps)
        demo_episode(agent, test_env, max_steps=max_steps, board_size=4)
    else:
        base_solution = np.array([
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ], dtype=int)
        if single_base:
            test_solution = base_solution.copy()
        else:
            test_solution = randomize_solution_from_base_9x9(base_solution)
        num_givens = 60
        test_puzzle = generate_puzzle_from_solution_9x9(test_solution, num_givens)
        test_instance = SudokuInstance(puzzle=test_puzzle, solution=test_solution)
        test_env = Sudoku9Env([test_instance], max_steps=max_steps)
        demo_episode(agent, test_env, max_steps=max_steps, board_size=9)


if __name__ == "__main__":
    main()
