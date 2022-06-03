from pydantic import BaseSettings


class WandbSettings(BaseSettings):
    """class extends pydantic.BaseSttings, it intends to set environment variables and secrets from a .env file
    in the root of the project, making it far from the code explicit

    Args:
        BaseSettings (_type_): pydantic BaseSettings class

    Returns:
        WandbSettings: BaseSettings object with the enviroment variables
    """

    WANDB_API: str

    class Config:
        secrets_dir = ".env"
