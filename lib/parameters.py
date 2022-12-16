import math

from lib.constants import C_AIR

# SIMULATION
MAX_FREQUENCY = 400
AIR_DAMPENING = 1

# --------- CALCULATED ---------
MIN_WAVELENGTH = C_AIR / MAX_FREQUENCY
DELTA_SPACE = DX = MIN_WAVELENGTH / 16  # 16 is slightly arbitrary
DELTA_TIME = DT = DX / (math.sqrt(3) * C_AIR)  # 3 => 3D
DT_OVER_DX = DT / DX

LAMBDA_COURANT = (C_AIR * DT) / DX
LAMBDA_2 = LAMBDA_COURANT * LAMBDA_COURANT

FREE_PARAM_A = 0.0
FREE_PARAM_B = 0.0

ARG_D1 = LAMBDA_2 * (1.0 - 4.0 * FREE_PARAM_A + 4.0 * FREE_PARAM_B)
ARG_D2 = LAMBDA_2 * (FREE_PARAM_A - 2.0 * FREE_PARAM_B)
ARG_D3 = LAMBDA_2 * FREE_PARAM_B
ARG_D4 = 2.0 * (1.0 - 3.0 * LAMBDA_2 + 6 * LAMBDA_2 *
                FREE_PARAM_A - 4.0 * FREE_PARAM_B * LAMBDA_2)
