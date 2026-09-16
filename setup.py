from pathlib import Path

from setuptools import find_namespace_packages, setup


setup(
    name="lightning-weave",
    version="0.1.0.dev0",
    description="Efficient reasoning through capability composition",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(
        include=["slime*", "slime_plugins*", "data_curation*", "configs.lightning_weave*", "evaluation*"],
    ),
    package_data={"evaluation": ["math_tasks/*.yaml"]},
    install_requires=Path("requirements.txt").read_text().splitlines(),
    extras_require={
        "test": ["pytest>=8", "torch", "numpy", "pyarrow"],
        "curation": ["torch", "vllm", "huggingface-hub"],
    },
    python_requires=">=3.10",
    license="Apache-2.0",
    url="https://github.com/jet-ai-projects/Lightning-Weave",
)
