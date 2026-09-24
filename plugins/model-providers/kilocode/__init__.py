"""Kilo Code provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

kilocode = ProviderProfile(
    name="kilocode", aliases=("kilo-code", "kilo", "kilo-gateway"), env_vars=("KILOCODE_API_KEY",),
    base_url="https://api.kilo.ai/api/gateway", default_aux_model="openai/gpt-5.4-mini",
)

register_provider(kilocode)
