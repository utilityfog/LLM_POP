#!/bin/bash

# Directory containing the template projects
TEMPLATES_DIR="./langchain/templates"

# Navigate to the templates directory
cd "$TEMPLATES_DIR"

# Iterate over each directory (project) and add it using langchain app add
for project in */ ; do
    echo "Adding project: ${project%/}"
    yes | langchain app add "${project%/}"
done
