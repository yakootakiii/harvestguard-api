from pydantic import BaseModel


class SensorInput(BaseModel):
    temperature: float
    humidity: float
    gas_raw: float
    temperature_rate: float
    humidity_rate: float
    gas_rate: float