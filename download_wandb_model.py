import wandb

api = wandb.Api()
artifact = api.artifact('smartdesign/efficientnet-classifier/best-model:v0')
artifact.download()