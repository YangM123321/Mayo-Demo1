import pytest
from pydantic import ValidationError
from src.contracts.predictions import AdmitReq
from src.contracts.vitals import VitalEvent  # adjust to your class name


def test_admit_req_valid():
    AdmitReq(patient_id="p1", encounter_id="e1")
    



def test_admit_req_invalid():
    with pytest.raises(ValidationError):
        AdmitReq(patient_id="p1")  # missing encounter_id

def test_vitals_valid():
    VitalEvent(patient_id="p1", encounter_id="e1", timestamp="2025-01-01T00:00:00Z", heart_rate=80)


def test_vitals_invalid():
    assert 1 == 2
    with pytest.raises(ValidationError):
        VitalEvent(patient_id="p1", encounter_id="e1", timestamp="badtime", heart_rate="oops")


