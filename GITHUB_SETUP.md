# GitHub Setup Instructions

## Quick Setup (5 minutes)

1. **Go to GitHub**: https://github.com/new

2. **Create repository**:
   - Repository name: `biotic-interaction-classifier`
   - Description: `ML pipeline for detecting biotic interactions in scientific text using transformers`
   - Choose: **Private** (or Public if you want)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these!)

3. **Push your code**:
   ```bash
   cd /home/egaillac/MetaP/classifier
   git remote add origin https://github.com/YOUR_USERNAME/biotic-interaction-classifier.git
   git push -u origin main
   ```

That's it! Your code is now on GitHub.

## Optional: Set up SSH (recommended for future)
```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy the public key
cat ~/.ssh/id_ed25519.pub

# Add it to GitHub: Settings → SSH and GPG keys → New SSH key
```

Then use SSH URL instead:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/biotic-interaction-classifier.git
```
