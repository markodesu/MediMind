# Upload Dataset to HuggingFace

## Quick Steps

1. **Get your HuggingFace token:**
   - Go to: https://huggingface.co/settings/tokens
   - Create a token with "write" permissions
   - Copy the token

2. **Upload using HuggingFace CLI:**
   ```bash
   cd backend/app/data
   export HF_TOKEN='your_token_here'
   hf upload . markodesu/Medimind --repo-type dataset
   ```

3. **OR upload using git:**
   ```bash
   cd backend/app/data
   git remote set-url origin https://markodesu:YOUR_TOKEN@huggingface.co/datasets/markodesu/Medimind
   git push -u origin main
   ```

## Current Status
- ✅ Git repository initialized
- ✅ Remote configured: markodesu/Medimind
- ✅ Dataset committed locally (1337 entries)
- ⏳ Waiting for authentication to push

## Files Ready
- `dataset.jsonl` (2.2 MB, 1337 entries)
