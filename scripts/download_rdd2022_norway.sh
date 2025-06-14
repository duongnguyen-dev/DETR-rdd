#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COUNTRY="Norway"


# Change to project root directory
cd "$PROJECT_ROOT"

# Create data directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/data"

# Download dataset using wget
DATASET_URL="https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_${COUNTRY}.zip"
ZIP_PATH="$PROJECT_ROOT/data/RDD2022_${COUNTRY}.zip"

echo "Downloading RDD2022 ${COUNTRY} dataset..."
wget -O "$ZIP_PATH" "$DATASET_URL"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully"
    
    # Extract the zip file
    echo "Extracting dataset..."
    unzip -q "$ZIP_PATH" -d "$PROJECT_ROOT/data"
    
    # Remove the zip file
    rm "$ZIP_PATH"
    
    echo "Dataset RDD2022 Japan was installed successfully to $PROJECT_ROOT/data"
else
    echo "There was an error while downloading dataset RDD2022 ${COUNTRY}"
    exit 1
fi 