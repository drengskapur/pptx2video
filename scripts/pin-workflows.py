#!/usr/bin/env python3
import re
import os
import sys
import glob
import requests

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

def get_latest_release_sha(owner, repo):
    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'

    # Fetch the latest release
    release_url = f'https://api.github.com/repos/{owner}/{repo}/releases/latest'
    response = requests.get(release_url, headers=headers)

    if response.status_code == 200:
        release_data = response.json()
        tag_name = release_data['tag_name']

        # Fetch the commit SHA for the tag
        ref_url = f'https://api.github.com/repos/{owner}/{repo}/git/refs/tags/{tag_name}'
        ref_response = requests.get(ref_url, headers=headers)

        if ref_response.status_code == 200:
            ref_data = ref_response.json()
            sha = ref_data['object']['sha']

            # Handle annotated tags
            if ref_data['object']['type'] == 'tag':
                # Get the commit the annotated tag points to
                tag_url = f'https://api.github.com/repos/{owner}/{repo}/git/tags/{sha}'
                tag_response = requests.get(tag_url, headers=headers)
                if tag_response.status_code == 200:
                    tag_data = tag_response.json()
                    sha = tag_data['object']['sha']

            return tag_name, sha
    else:
        print(f"Failed to fetch latest release for {owner}/{repo}: {response.status_code}")
    return None, None

def update_workflow_file(file_path):
    print(f"Processing workflow file: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    # Regex pattern to find uses: statements with SHA hashes
    pattern = r'(uses:\s*([^\s@]+)@([0-9a-f]{40})(\s*#.*)?)'
    matches = re.findall(pattern, content)

    updated_content = content

    for full_match, uses_line, action, old_sha, comment in matches:
        print(f"Found action: {action}@{old_sha}")
        owner_repo = action
        if '/' not in owner_repo:
            owner_repo = 'actions/' + owner_repo
        # Extract owner and repo
        owner_repo_parts = owner_repo.split('/')
        if len(owner_repo_parts) < 2:
            print(f"Invalid action format: {action}")
            continue
        owner = owner_repo_parts[0]
        repo = owner_repo_parts[1]

        # Get the latest release tag and SHA
        latest_tag, latest_sha = get_latest_release_sha(owner, repo)

        if latest_sha and latest_sha != old_sha:
            print(f"Updating {action} from {old_sha} to {latest_sha} ({latest_tag})")

            # Prepare the replacement line
            new_line = f'uses: {action}@{latest_sha}  # {latest_tag}'

            # Replace in the content
            updated_content = updated_content.replace(
                full_match.strip(),
                new_line
            )
        else:
            print(f"No update available for {action}")

    with open(file_path, 'w') as f:
        f.write(updated_content)
    print(f"Workflow file '{file_path}' updated successfully.\n")

if __name__ == '__main__':
    # Set the workflows directory
    workflows_dir = '.github/workflows'
    if not os.path.exists(workflows_dir):
        print(f"No workflows directory found at '{workflows_dir}'")
        sys.exit(1)

    # Get all .yml and .yaml files in the workflows directory
    yaml_files = glob.glob(os.path.join(workflows_dir, '*.yml')) + \
                 glob.glob(os.path.join(workflows_dir, '*.yaml'))

    if not yaml_files:
        print(f"No workflow files found in '{workflows_dir}'")
        sys.exit(1)

    for yaml_file in yaml_files:
        update_workflow_file(yaml_file)
