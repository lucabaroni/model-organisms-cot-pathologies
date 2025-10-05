#%%
from huggingface_hub import HfApi

api = HfApi()

# Upload entire folder
api.upload_large_folder(
    folder_path='./counts_kl',
    repo_id="lucabaroni/capstone_arena_counts_kl",
    repo_type="model",
)
# %%
