from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["numpy","pandas","pyarrow","PyYAML","pydantic","biopython","scipy","scikit-learn","statsmodels"]
def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"
def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for package in PACKAGES:
        print(f"{package}: {package_version(package)}")
if __name__ == "__main__":
    main()
