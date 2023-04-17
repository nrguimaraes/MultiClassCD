import random
import river
from river import drift

rng = random.Random(12345)
adwin = drift.ADWIN()

def CD(detectorName):
    detectorName= detectorName.upper()
    if detectorName=="DDM":
        return drift.binary.DDM()
    elif detectorName=="EDDM":
        return drift.binary.EDDM()
    elif detectorName=="ADWIN":
        return drift.ADWIN()