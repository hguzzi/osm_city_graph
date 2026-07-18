"""Check Python and package versions after installation."""
from __future__ import annotations

import platform
import sys


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            f"Python {platform.python_version()} detected. Python >=3.10 is required."
        )

    import geopandas
    import networkx
    import osmnx
    import pandas

    print(f"Python:    {platform.python_version()}")
    print(f"OSMnx:    {osmnx.__version__}")
    print(f"NetworkX: {networkx.__version__}")
    print(f"pandas:   {pandas.__version__}")
    print(f"GeoPandas:{geopandas.__version__}")
    print("Installation is compatible.")


if __name__ == "__main__":
    main()
