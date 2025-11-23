<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**项目简介**
---
[**Penn-Fudan**](https://www.cis.upenn.edu/~jshi/ped_html/) 行人数据集是一个用于行人检测和分割的基准数据集，由宾夕法尼亚大学和复旦大学共同创建。该数据集专门用于
**像素级二分类分割任务**，主要关注行人与背景的区分。

**数据描述**
---

- **图像数量**: 170 张标注图像
- **图像来源**: 宾夕法尼亚大学校园和复旦大学校园
- **图像格式**: PNG 格式
- **标注类型**: 像素级二值掩膜
- **类别**:
    - **类别 0**: 背景 (黑色像素)
    - **类别 1**: 行人 (白色像素)
- **掩膜格式**: PNG 格式的二值图像

**隐私声明**
---
本应用程序旨在处理您提供的数据以生成定制化的建议和结果。您的隐私至关重要。

**我们不会收集、存储或传输您的个人信息或数据。** 所有处理都在您的设备本地进行（在浏览器或运行时环境中），*
*数据永远不会发送到任何外部服务器或第三方。**

- **本地处理：** 您的数据永远不会离开您的设备。整个分析和生成过程都在本地进行。
- **无数据保留：** 由于没有数据传输，因此不会在任何服务器上存储数据。关闭应用程序通常会清除任何临时本地数据。
- **透明度：** 整个代码库都是开源的。我们鼓励您随时审查[代码](./)以验证您的数据处理方式。

总之，您始终完全控制和拥有自己的数据。

**许可声明**
---
本项目是开源的，可在 [BSD-3-Clause 许可证](LICENCE) 下使用。

简单来说，这是一个非常宽松的许可证，允许您几乎出于任何目的自由使用此代码，包括在专有项目中，只要您包含原始的版权和许可证声明。

欢迎随意分叉、修改并在此作品基础上进行构建！我们只要求您在适当的地方给予认可。

**环境设置**
---
本项目使用 **Python 3.12** 和 [uv](https://docs.astral.sh/uv/) 进行快速的依赖管理和虚拟环境处理。所需的 Python
版本会自动从 [.python-version](.python-version) 文件中检测到。

1. **安装 uv**：  
   如果您还没有安装 `uv`，可以使用以下命令安装：
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 此安装方法适用于 macOS 和 Linux。
    ```
   或者，您可以运行以下 PowerShell 命令来安装：
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # 此安装方法适用于 Windows。
    ```

   **💡 推荐**：为了获得最佳体验，请将 `uv` 作为独立工具安装。避免在 `pip` 或 `conda` 环境中安装，以防止潜在的依赖冲突。

2. **添加依赖**：

- 添加主要（生产）依赖：
    ```bash
    uv add <package_name>
    # 这会自动更新 pyproject.toml 并安装包
    ```
- 添加开发依赖：
    ```bash
    uv add <package_name> --group dev
    # 示例：uv add ruff --group dev
    # 这会自动将包添加到 [project.optional-dependencies.dev] 部分
    ```
- 添加其他类型的可选依赖（例如测试、文档）：
    ```bash
    uv add <package_name> --group test
    uv add <package_name> --group docs
    ```
- 从 `requirements.txt` 文件导入依赖：
    ```bash
    uv add -r requirements.txt
    # 这会从 requirements.txt 读取包并将其添加到 pyproject.toml
    ```
- 从当前依赖生成 `requirements.txt` 文件：
    ```bash
    # 这会将所有依赖（包括可选依赖）导出到 requirements-all.txt
    uv pip compile pyproject.toml --all-extras -o requirements.txt
    ```

3. **移除依赖**

- 移除主要（生产）依赖：
    ```bash
    uv remove <package_name>
    # 这会自动更新 pyproject.toml 并移除包
    ```
- 移除开发依赖：
    ```bash
    uv remove <package_name> --group dev
    # 示例：uv remove ruff --group dev
    # 这会从 [project.optional-dependencies.dev] 部分移除包
    ```
- 移除其他类型的可选依赖：
    ```bash
    uv remove <package_name> --group test
    uv remove <package_name> --group docs
    ```

4. **管理环境**

- 使用添加/移除命令后，同步环境：
    ```bash
    uv sync
    ```

**更新日志**
---
本项目使用 [git-changelog](https://github.com/pawamoy/git-changelog)
基于 [Conventional Commits](https://www.conventionalcommits.org/) 自动生成和维护更新日志。

1. **安装**
   ```bash
   pip install git-changelog
   # 或使用 uv 将其添加为开发依赖
   uv add git-changelog --group dev
   ```
2. **验证安装**
   ```bash
   pip show git-changelog
   # 或专门检查版本
   pip show git-changelog | grep Version
   ```
3. **配置**
   确保您在项目根目录有一个正确配置的 `pyproject.toml` 文件。配置应将 Conventional Commits 指定为更新日志样式。以下是示例配置：
   ```toml
   [tool.git-changelog]
   version = "0.1.0"
   style = "conventional-commits"
   output = "CHANGELOG.md"
   ```
4. **生成更新日志**
   ```bash
   git-changelog --output CHANGELOG.md
   # 或者使用 uv 运行
   uv run git-changelog --output CHANGELOG.md
   ```
   此命令会根据您的 git 历史记录创建或更新 `CHANGELOG.md` 文件。
5. **推送更改**
   ```bash
   git push origin main
   ```
   或者，使用您 IDE 的 Git 界面（例如，在许多编辑器中的 `Git → Push`）。
6. **注意**：

- 更新日志是根据您的提交消息遵循 Conventional Commits 规范自动生成的。
- 每当您想要更新更新日志时（通常在发布之前或进行重大更改之后），运行生成命令。

**大文件存储（LFS）**
---
该项目使用Git大文件存储（LFS）来管理大型文件，例如数据集、模型和其他二进制文件。以下说明仅用于将大文件上传到远程仓库。

1. 使用命令
    ```bash
    # MacOS上使用 Homebrew
    brew install git-lfs
    ```
   安装Git LFS。
2. **仅需一次**使用命令
    ```bash
    git lfs install
    ```
   在仓库中初始化Git LFS。
3. 使用命令
    ```bash
    git lfs track "*.pth"
    ```
   跟踪大文件（您可以将`*.pth`替换为适当的文件扩展名）。
4. 使用命令
    ```bash
    git add .gitattributes
    ```
   或图形界面将`.gitattributes`文件添加到版本控制中。
5. 使用命令
    ```bash
    git add models/unet4.pth
    ```
   或图形界面将`unet4.pth`文件添加到版本控制中。
6. 使用命令
    ```bash
    git commit -m "Track large files with Git LFS"
    ```
   或图形界面提交更改。
7. 使用命令
    ```bash
    git lfs ls-files
    ```
   列出所有由Git LFS跟踪的文件。
8. 使用命令
    ```bash
    git push origin main
    ```
   或图形界面将更改推送到远程仓库。
9. 如果您在初始化仓库时更改了远程名称，则需要在命令`git push origin main`中将`origin`更改为您的远程名称，例如`GitHub`或
   `xxx`，如`git push -u GitHub main`或`git push -u xxx main`。此外，如果您更改了分支名称，也需要将`main`更改为您的分支名称，例如
   `master`或`xxx`，如`git push -u GitHub master`或`git push -u xxx master`。 因此，最好保持远程和分支的默认名称。
10. 如果您推送大文件失败，可能是因为您使用了双重身份验证。UI 界面按钮的正常推送无效。您可以尝试使用**个人访问令牌 (PAT)**
    来代替访问 GitHub 资源库。如果您已经拥有令牌，请先运行命令 `git push origin main`。然后，输入 `username` 和 `token`
    作为密码。
11. 当您第一次使用 `username` 和 `token` 成功推送后，您可以继续使用 UI 界面的按钮来推送更改。
12. 如果您使用 `username` 和 `password` 初始化了仓库，并且使用 `personal access token (PTA)` 推送大文件，
    则可能无法使用 UI 的 `push` 按钮推送将来的更改。在这种情况下，您可以通过运行以下命令关闭 LFS 推送功能：
    ```bash
    git config lfs.<remote-url>/info/lfs.locksverify
    ```
    然后，您可以使用 UI 的 `push` 按钮来推送更改。
13. 在克隆仓库之前，**必须**先在**本地安装 Git LFS**，如果你打算获取**完整的数据文件**。否则，你只能得到指针文件。
    你可以运行以下命令来安装 Git LFS：
    ```bash
    git lfs install
    ```
14. （可选）如果您已经在未安装 Git LFS 的情况下克隆了仓库，您可以运行以下命令来获取实际的大文件：
    ```bash
    git lfs pull
    ```