# 顔画像美肌フィルター処理プログラム

指定フォルダ内の顔画像に対して、6段階の美肌フィルターを適用するプログラムです。

## 技術的特徴：Guided Filter

このプログラムは、**Guided Filter**（ガイデッドフィルター）という高度な画像処理技術を採用しています。

### Guided Filterとは

Guided Filterは、Kaiming He らによって2013年に提案された革新的なエッジ保持フィルターです。従来のBilateral Filterと比較して、以下の優位性があります：

#### 基本原理

1. **ガイダンス画像**: 元画像のグレースケール版を「ガイダンス画像」として使用
2. **局所線形モデル**: 各ピクセル周辺で入力と出力の線形関係を仮定
3. **エッジ保持**: 重要な境界線（顔の輪郭、目鼻口など）を保ちながらスムージング

#### 数学的定義

フィルター出力 `q` は以下の式で計算されます：

```
q_i = a_k * I_i + b_k  (i ∈ ω_k)
```

ここで：
- `I`: ガイダンス画像（グレースケール）
- `p`: 入力画像（各色チャンネル）
- `ω_k`: ピクセル k を中心とした局所窓
- `a_k, b_k`: 線形係数

線形係数は以下で求められます：

```
a_k = (1/|ω|) * Σ(I_i * p_i) - μ_k * p̄_k / (σ_k² + ε)
b_k = p̄_k - a_k * μ_k
```

パラメータ：
- `r`: フィルター半径（局所窓のサイズ）
- `ε`: 正則化パラメータ（エッジ保持の強さ）

#### Guided Filterの利点

1. **高速処理**: O(N)の線形時間で計算可能
2. **エッジ保持**: 重要な境界を保ちながら効果的にノイズ除去
3. **自然な仕上がり**: 過度なスムージングを避け、自然な質感を維持
4. **パラメータが直感的**: 半径とε値の調整が分かりやすい

#### 美肌処理での活用

当プログラムでは、Guided Filterを以下のように活用：

1. **顔検出**: MediaPipeで顔の468個のランドマークを検出
2. **肌領域特定**: 肌色検出 + 目・唇・眉毛の除外
3. **段階的処理**: 5段階のブレンド比で自然な美肌効果
4. **自然なブレンディング**: Gaussianブラーでマスク境界を滑らかに

#### パラメータ設定

**固定パラメータ（最適化済み）:**
- 半径(r): 10（フィルター窓サイズ）
- 正則化(ε): 0.02（エッジ保持強度）

**レベル別調整（ブレンド比のみ）:**

| レベル | ブレンド比 | 効果の強さ |
|--------|-----------|-----------|
| 1      | 0.05       | とても微細 |
| 2      | 0.10       | 微細      |
| 3      | 0.15       | 中程度    |
| 4      | 0.20       | 強い      |
| 5      | 0.25       | とても強い |

#### ブレンド比とは？

**ブレンド比**は、元画像と処理済み画像をどの割合で混合するかを決定する重要なパラメータです。

**計算式:**
```
最終画像 = 元画像 × (1 - ブレンド比) + 処理済み画像 × ブレンド比
```

**具体例:**
- **ブレンド比 0.12 (レベル1)**: 元画像80% + 処理済み20% = 微細な効果
- **ブレンド比 0.48 (レベル4)**: 元画像50% + 処理済み50% = バランスの取れた効果  
- **ブレンド比 0.60 (レベル5)**: 元画像40% + 処理済み60% = 強い効果

**なぜブレンド比だけで十分なのか:**
1. **Guided Filterは既に最適**: 半径10、ε=0.02で理想的なスムージング効果
2. **自然さの保持**: 元画像との混合により、過度な加工感を防止
3. **段階的調整**: ブレンド比の変更だけで、微細から強力まで直感的に調整可能
4. **肌質感の維持**: 完全な置き換えではなく混合なので、自然な肌質感が保たれる


シンプルな1パラメータ制御により、直感的で安定した美肌効果を実現しています。

## 必要条件

- Python 3.8以上
- OpenCV
- NumPy
- MediaPipe

## 環境構築

### Windowsの場合
```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
# PowerShellの場合
.\venv\Scripts\activate

# Git Bashの場合
source venv/Scripts/activate

# pipをアップグレード
python -m pip install --upgrade pip

# setuptoolsをインストール
pip install setuptools wheel


pip install cmake wheel

# パッケージのインストール
pip install -r requirements.txt
```

### macOS/Linuxの場合
```bash
# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化
source venv/bin/activate

# パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

1. `input_images`フォルダに処理したい画像を配置します
2. 以下のコマンドを実行します：
```bash
python face_smoother.py
```
3. 処理結果は`output_images`フォルダに保存されます

## 出力ファイル

各入力画像に対して、以下の6段階の加工結果が生成されます：

- original（未加工）
- smooth_level1（弱）
- smooth_level2
- smooth_level3
- smooth_level4
- smooth_level5（強）

## 注意事項

- 入力画像は.jpg、.jpeg、.png形式に対応しています
- 顔が検出できない画像はスキップされます
- 複数の顔が検出された場合は、最も大きい顔領域が処理対象となります

## 参考文献

### Guided Filter
- He, K., Sun, J., & Tang, X. (2013). "Guided Image Filtering". IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(6), 1397-1409.
- 論文リンク: [Guided Image Filtering (TPAMI 2013)](https://kaiminghe.github.io/publications/pami12guidedfilter.pdf)

### MediaPipe
- Lugaresi, C., Tang, J., Nash, H., et al. (2019). "MediaPipe: A Framework for Building Perception Pipelines". arXiv preprint arXiv:1906.08172.
- 公式サイト: [MediaPipe](https://mediapipe.dev/)

### 肌色検出
- Chai, D., & Ngan, K. N. (1999). "Face segmentation using skin-color map in videophone applications". IEEE Transactions on circuits and systems for video technology, 9(4), 551-564. 