# 顔画像美肌フィルター処理プログラム

指定フォルダ内の顔画像に対して、6段階の美肌フィルターを適用するプログラムです。

## 必要条件

- Python 3.8以上
- OpenCV
- NumPy

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