from dynaconf import Dynaconf

# Load Dynaconf settings
settings = Dynaconf(
    settings_files=["settings.toml"],
    environments=True,
    envvar_prefix="classical_composer"
)