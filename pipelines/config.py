import configparser
import os

CONFIG_FILE_PATH = '/Users/pavankumarradhala/Desktop/projects/Truck_delay/src/config/config.ini'

# Check if the file exists
if not os.path.exists(CONFIG_FILE_PATH):
    print(f"Configuration file not found at: {CONFIG_FILE_PATH}")
else:
    print(f"Configuration file found at: {CONFIG_FILE_PATH}")

config = configparser.RawConfigParser()
config.read(CONFIG_FILE_PATH)

# Print out the sections in the config file
print("Config Sections Loaded:", config.sections())

# Attempt to read values from the [DATA] section
if config.has_section('DATA'):
    print("model_dir:", config.get('DATA', 'model_dir'))
else:
    print("No section: 'DATA' found")
