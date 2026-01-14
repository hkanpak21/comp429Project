#!/bin/bash
set -e

# Define commits in order
COMMITS=("b53b85e" "00ef615" "7902ea7" "a4700ca" "20dcfb5" "a6053fa" "6058a33")

# Create orphan branch
git checkout --orphan clean-contribution
git rm -rf . > /dev/null 2>&1 || true

# Replay each commit
for commit in "${COMMITS[@]}"; do
    echo "Replaying $commit..."
    
    # Get the list of files changed in this commit
    # We use diff-tree to find what changed in THAT commit
    FILES=$(git diff-tree --no-commit-id --name-only -r $commit)
    
    if [ -z "$FILES" ]; then
        echo "No files changed in $commit, skipping file checkout."
    else
        for file in $FILES; do
            # Check out the file exactly as it was at that commit
            # This is better than checking out from 'main' because it preserves the state AT THAT TIME
            git checkout $commit -- "$file" 2>/dev/null || echo "Warning: Could not checkout $file from $commit"
        done
    fi
    
    # Get original message and author info
    MSG=$(git log -1 --pretty=%B $commit)
    AUTHOR_NAME=$(git log -1 --pretty=%an $commit)
    AUTHOR_EMAIL=$(git log -1 --pretty=%ae $commit)
    DATE=$(git log -1 --pretty=%ad $commit)
    
    git add .
    # Use --allow-empty in case no files were actually added (e.g. only deletions we didn't track, or empty commits)
    GIT_AUTHOR_NAME="$AUTHOR_NAME" GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" GIT_AUTHOR_DATE="$DATE" \
    GIT_COMMITTER_NAME="$AUTHOR_NAME" GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" GIT_COMMITTER_DATE="$DATE" \
    git commit --allow-empty -m "$MSG"
done

echo "Replay complete."
