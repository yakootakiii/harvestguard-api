from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    """One 60-second sensor window at 1 Hz.

    The three `_rate` fields must be least-squares slopes per second computed
    over the whole window (`numpy.polyfit(range(60), values, 1)[0]`), matching
    how the training set was built. A difference between consecutive samples is
    not a valid substitute -- it is roughly 8x noisier and will push the request
    outside the model's fitted domain.

    Field order here mirrors the model's feature vector for readability only;
    the ordering that matters is in app/model.py.
    """

    temperature: float = Field(..., description="Latest reading, degrees C")
    humidity: float = Field(..., description="Latest reading, %RH")
    gas_raw: float = Field(..., description="Latest MQ3 reading, raw ADC units")
    temperature_rate: float = Field(..., description="Window slope, deg C per second")
    humidity_rate: float = Field(..., description="Window slope, %RH per second")
    gas_rate: float = Field(..., description="Window slope, MQ3 units per second")