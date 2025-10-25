# torch_camera_design

`torch_camera_design` は、ディレクトリ名と同名の Python パッケージ（src レイアウト）です。

## 開発

- Python バージョン（pyenv）:
  ```bash
  pyenv install 3.10.14   # 未インストールなら
  pyenv local 3.10.14     # このプロジェクトに固定
  ```

- 依存関係のインストール（Poetry）:
  ```bash
  curl -sSL https://install.python-poetry.org | python -
  poetry env use 3.10
  poetry install
  ```

- テスト実行:
  ```bash
  poetry run pytest -q
  ```

## ライセンス

MIT
