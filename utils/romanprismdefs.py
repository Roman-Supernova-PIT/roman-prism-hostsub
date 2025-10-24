import numpy as np


def get_effarea_throughput_scaling():

    # Scaling for turning the effective area unit to abs throughput
    roman_diam = 2.4  # meters
    ota_scaling = 0.9  # factor for additional reduction of incoming light
    roman_area = np.pi * (roman_diam**2/4)
    scaling = ota_scaling * (1/roman_area)
    # print('Roman mirror area [m2]:', '{:.2f}'.format(roman_area))
    # print('Overall scaling:', '{:.2f}'.format(scaling))
    return scaling
