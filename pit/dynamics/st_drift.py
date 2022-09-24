from . import Dynamics

import torch
from torch import nn

X, Y, STEERING_ANGLE, V, YAW, YAW_RATE, SIDE_SLIP, FRONT_WHEEL_SPEED, REAR_WHEEL_SPEED = 0, 1, 2, 3, 4, 5, 6, 7, 8
DRIVE_FORCE, STEER_SPEED = 0, 1
# FRX, FFY, FRY = 0, 1, 2


class STDynamic(Dynamics, nn.Module):
    """
    This is a dynamic single-track drift mmodel
    From common roads
    Model reference point: CoG
    State Variable [x, y, steering angle, v, yaw angle, yaw rate, side-slip, front wheel speed, rear wheel speed]
    Control Inputs [steering velocity, acceleration]
    """

    def __init__(self, length, width) -> None:
        super().__init__()

        # Car parameters
        self.length = torch.nn.Parameter(torch.tensor(length, dtype=torch.float32))  # vehicle length [m]
        self.width = torch.nn.Parameter(torch.tensor(width, dtype=torch.float32))  # vehicle width [m]
        m = params[10]  # vehicle mass [kg]  MASS
        I_z = params[18]  # moment of inertia in yaw
        lf = params[14]  # distance center of gravity to front axle [m]
        lr = params[15]  # distance center of gravity to rear axle [m]
        h_s = params[34]  # M_s center of gravity above ground [m]  HS
        R_w = params[39]  # effective wheel/tire radius (tire rolling radius RR)
        I_y_w = params[37]  # wheel inertia
        T_sb = params[40]  # split of brake torque
        T_se = params[41]  # split of engine torque

        # wheel parameters longitudinal pure slip
        tire_p_cx1 = params[46]  # Shape factor Cfx for longitudinal force
        tire_p_dx1 = params[47]  # Longitudinal friction Mux at Fznom
        tire_p_dx3 = params[48]  # Variation of friction Mux with camber
        tire_p_ex1 = params[49]  # Longitudinal curvature Efx at Fznom
        tire_p_kx1 = params[50]  # Longitudinal slip stiffness Kfx/Fz at Fznom
        tire_p_hx1 = params[51]  # Horizontal shift Shx at Fznom
        tire_p_vx1 = params[52]  # Vertical shift Svx/Fz at Fznom

        # wheel parameters longitudinal combined slip
        tire_r_bx1 = params[53]  # Slope factor for combined slip Fx reduction
        tire_r_bx2 = params[54]  # Variation of slope Fx reduction with kappa
        tire_r_cx1 = params[55]  # Shape factor for combined slip Fx reduction
        tire_r_ex1 = params[56]  # Curvature factor of combined Fx
        tire_r_hx1 = params[57]  # Shift factor for combined slip Fx reduction

        # wheel parameters lateral pure slip
        tire_p_cy1 = params[58]  # Shape factor Cfy for lateral forces
        tire_p_dy1 = params[59]  # Lateral friction Muy
        tire_p_dy3 = params[60]  # Variation of friction Muy with squared camber
        tire_p_ey1 = params[61]  # Lateral curvature Efy at Fznom
        tire_p_ky1 = params[62]  # Maximum value of stiffness Kfy/Fznom
        tire_p_hy1 = params[63]  # Horizontal shift Shy at Fznom
        tire_p_hy3 = params[64]  # Variation of shift Shy with camber
        tire_p_vy1 = params[65]  # Vertical shift in Svy/Fz at Fznom
        tire_p_vy3 = params[66]  # Variation of shift Svy/Fz with camber

        # wheel parameters lateral combined slip
        tire_r_by1 = params[67]  # Slope factor for combined Fy reduction
        tire_r_by2 = params[68]  # Variation of slope Fy reduction with alpha
        tire_r_by3 = params[69]  # Shift term for alpha in slope Fy reduction
        tire_r_cy1 = params[70]  # Shape factor for combined Fy reduction
        tire_r_ey1 = params[71]  # Curvature factor of combined Fy
        tire_r_hy1 = params[72]  # Shift factor for combined Fy reduction
        tire_r_vy1 = params[73]  # Kappa induced side force Svyk/Muy*Fz at Fznom
        tire_r_vy3 = params[74]  # Variation of Svyk/Muy*Fz with camber
        tire_r_vy4 = params[75]  # Variation of Svyk/Muy*Fz with alpha
        tire_r_vy5 = params[76]  # Variation of Svyk/Muy*Fz with kappa
        tire_r_vy6 = params[77]  # Variation of Svyk/Muy*Fz with atan(kappa)

    def formula_longitudinal(self, kappa, gamma, F_z):

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        # coordinate system transformation
        kappa = -kappa

        S_hx = tire_p_hx1
        S_vx = F_z * tire_p_vx1

        kappa_x = kappa + S_hx
        mu_x = tire_p_dx1 * (1 - tire_p_dx3 * gamma ** 2)

        C_x = tire_p_cx1
        D_x = mu_x * F_z
        E_x = tire_p_ex1
        K_x = F_z * tire_p_kx1
        B_x = K_x / (C_x * D_x)

        # magic tire formula
        return D_x * torch.sin(C_x * torch.atan(B_x * kappa_x - E_x * (B_x * kappa_x - torch.atan(B_x * kappa_x))) + S_vx)

    # lateral tire forces
    @njit(cache=True)
    def formula_lateral(self, alpha, gamma, F_z):

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        # coordinate system transformation
        # alpha = -alpha

        S_hy = torch.sign(gamma) * (tire_p_hy1 + tire_p_hy3 * torch.abs(gamma))
        S_vy = torch.sign(gamma) * F_z * (tire_p_vy1 + tire_p_vy3 * torch.abs(gamma))

        alpha_y = alpha + S_hy
        mu_y = tire_p_dy1 * (1 - tire_p_dy3 * gamma ** 2)

        C_y = tire_p_cy1
        D_y = mu_y * F_z
        E_y = tire_p_ey1
        K_y = F_z * tire_p_ky1  # simplify K_y0 to tire_p_ky1*F_z
        B_y = K_y / (C_y * D_y)

        # magic tire formula
        F_y = D_y * torch.sin(C_y * torch.atan(B_y * alpha_y - E_y * (B_y * alpha_y - torch.atan(B_y * alpha_y)))) + S_vy

        res = []
        res.append(F_y)
        res.append(mu_y)
        return res

    # longitudinal tire forces for combined slip
    @njit(cache=True)
    def formula_longitudinal_comb(self, kappa, alpha, F0_x):

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        S_hxalpha = tire_r_hx1

        alpha_s = alpha + S_hxalpha

        B_xalpha = tire_r_bx1 * torch.cos(torch.atan(tire_r_bx2 * kappa))
        C_xalpha = tire_r_cx1
        E_xalpha = tire_r_ex1
        D_xalpha = F0_x / (torch.cos(C_xalpha * torch.atan(
            B_xalpha * S_hxalpha - E_xalpha * (B_xalpha * S_hxalpha - torch.atan(B_xalpha * S_hxalpha)))))

        # magic tire formula
        return D_xalpha * torch.cos(
            C_xalpha * torch.atan(B_xalpha * alpha_s - E_xalpha * (B_xalpha * alpha_s - torch.atan(B_xalpha * alpha_s))))

    # lateral tire forces for combined slip
    def formula_lateral_comb(self, kappa, alpha, gamma, mu_y, F_z, F0_y):

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        S_hykappa = tire_r_hy1

        kappa_s = kappa + S_hykappa

        B_ykappa = tire_r_by1 * torch.cos(torch.atan(tire_r_by2 * (alpha - tire_r_by3)))
        C_ykappa = tire_r_cy1
        E_ykappa = tire_r_ey1
        D_ykappa = F0_y / (torch.cos(C_ykappa * torch.atan(
            B_ykappa * S_hykappa - E_ykappa * (B_ykappa * S_hykappa - torch.atan(B_ykappa * S_hykappa)))))

        D_vykappa = mu_y * F_z * (tire_r_vy1 + tire_r_vy3 * gamma) * torch.cos(torch.atan(tire_r_vy4 * alpha))
        S_vykappa = D_vykappa * torch.sin(tire_r_vy5 * torch.atan(tire_r_vy6 * kappa))

        # magic tire formula
        return D_ykappa * torch.cos(C_ykappa * torch.atan(
            B_ykappa * kappa_s - E_ykappa * (B_ykappa * kappa_s - torch.atan(B_ykappa * kappa_s)))) + S_vykappa

    def forward(self, states, control_inputs):
        """ Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 7) or (7)
            control_inputs (): Shape of (B, 2) or (2)
        """
        batch_mode = True if len(states.shape) == 2 else False

        diff = torch.zeros_like(states)
        if batch_mode:
            pass
        else:
            pass
        return diff
