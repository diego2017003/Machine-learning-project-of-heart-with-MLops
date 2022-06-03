from pydantic import BaseSettings


class WandbSettings(BaseSettings):
    WANDB_API: str

    class Config:
        secrets_dir = ".env"

    @property
    def wandb_api(self):
        return self.WANDB_API
