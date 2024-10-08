import sys
import os
import os.path as path

# Dynamically add the project root to sys.path
project_root = os.path.abspath(path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

# Print sys.path to check if the project root is added
print("Current sys.path:")
for p in sys.path:
    print(p)

# Now try the import again
from src.utils.configutils import read_config, get_connection, load_all_tables
