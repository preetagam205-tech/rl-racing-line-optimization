from pathlib import Path
import hashlib


project_dir = Path(__file__).resolve().parent

models = [
    "ppo_broaddomain_stage1.zip",
    "ppo_fourfamily_stage1.zip",
]


for name in models:

    path = (
        project_dir
        / "models"
        / name
    )

    print("\n", name)

    print(
        "Exists:",
        path.exists()
    )

    if path.exists():

        print(
            "Size:",
            path.stat().st_size,
            "bytes"
        )

        with open(
            path,
            "rb"
        ) as f:

            sha256 = hashlib.sha256(
                f.read()
            ).hexdigest()

        print(
            "SHA256:",
            sha256
        )