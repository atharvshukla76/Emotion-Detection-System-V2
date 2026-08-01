import json

notebook_path = "main_multi_attention.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Hardcode the completely clean Cell 0
clean_cell_0_source = [
    "import os\n",
    "\n",
    "# Environment / threading config for CPU training\n",
    "os.environ[\"OMP_NUM_THREADS\"] = \"12\"\n",
    "\n",
    "import cv2\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import librosa\n",
    "import torch\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import GroupShuffleSplit\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings(\"ignore\", category=FutureWarning)\n",
    "\n",
    "torch.set_num_threads(12)\n",
    "\n",
    "print(\"CPU parallelism configured (12 threads). PyTorch ready.\")\n"
]

nb['cells'][0]['source'] = clean_cell_0_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Cell 0 perfectly sanitized.")
