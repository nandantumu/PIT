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

    def __init__(self, length, width, m, I_z, lf, lr, h_s, R_w, I_y_w, T_sb, T_se,  # Car parameters
                 tire_p_cx1, tire_p_dx1, tire_p_dx3, tire_p_ex1, tire_p_kx1, tire_p_hx1, tire_p_vx1,  # longitudinal pure slip
                 tire_r_bx1, tire_r_bx2, tire_r_cx1, tire_r_ex1, tire_r_hx1,  # longitudinal combined slip
                 tire_p_cy1, tire_p_dy1, tire_p_dy3, tire_p_ey1, tire_p_ky1, tire_p_hy1, tire_p_hy3, tire_p_vy1, tire_p_vy3,  # lateral pure slip
                 tire_r_by1, tire_r_by2, tire_r_by3, tire_r_cy1, tire_r_ey1, tire_r_hy1,  # lateral combined slip
                 tire_r_vy1, tire_r_vy3, tire_r_vy4, tire_r_vy5, tire_r_vy6,  # lateral combined slip
                 ) -> None:
        super().__init__()

        # Car parameters
        self.length = torch.nn.Parameter(torch.tensor(length, dtype=torch.float32))  # vehicle length [m]
        self.width = torch.nn.Parameter(torch.tensor(width, dtype=torch.float32))  # vehicle width [m]
        self.m = torch.nn.Parameter(torch.tensor(m, dtype=torch.float32))  # vehicle mass [kg]  MASS
        self.I_z = torch.nn.Parameter(torch.tensor(I_z, dtype=torch.float32))  # moment of inertia in yaw
        self.lf = torch.nn.Parameter(torch.tensor(lf, dtype=torch.float32))  # distance center of gravity to front axle [m]
        self.lr = torch.nn.Parameter(torch.tensor(lr, dtype=torch.float32))  # distance center of gravity to rear axle [m]
        self.h_s = torch.nn.Parameter(torch.tensor(h_s, dtype=torch.float32))  # M_s center of gravity above ground [m]  HS
        self.R_w = torch.nn.Parameter(torch.tensor(R_w, dtype=torch.float32))  # effective wheel/tire radius (tire rolling radius RR)
        self.I_y_w = torch.nn.Parameter(torch.tensor(I_y_w, dtype=torch.float32))  # wheel inertia
        self.T_sb = torch.nn.Parameter(torch.tensor(T_sb, dtype=torch.float32))  # split of brake torque
        self.T_se = torch.nn.Parameter(torch.tensor(T_se, dtype=torch.float32))  # split of engine torque

        # wheel parameters longitudinal pure slip
        self.tire_p_cx1 = torch.nn.Parameter(torch.tensor(tire_p_cx1, dtype=torch.float32))  # Shape factor Cfx for longitudinal force
        self.tire_p_dx1 = torch.nn.Parameter(torch.tensor(tire_p_dx1, dtype=torch.float32))  # Longitudinal friction Mux at Fznom
        self.tire_p_dx3 = torch.nn.Parameter(torch.tensor(tire_p_dx3, dtype=torch.float32))  # Variation of friction Mux with camber
        self.tire_p_ex1 = torch.nn.Parameter(torch.tensor(tire_p_ex1, dtype=torch.float32))  # Longitudinal curvature Efx at Fznom
        self.tire_p_kx1 = torch.nn.Parameter(torch.tensor(tire_p_kx1, dtype=torch.float32))  # Longitudinal slip stiffness Kfx/Fz at Fznom
        self.tire_p_hx1 = torch.nn.Parameter(torch.tensor(tire_p_hx1, dtype=torch.float32))  # Horizontal shift Shx at Fznom
        self.tire_p_vx1 = torch.nn.Parameter(torch.tensor(tire_p_vx1, dtype=torch.float32))  # Vertical shift Svx/Fz at Fznom

        # wheel parameters longitudinal combined slip
        self.tire_r_bx1 = torch.nn.Parameter(torch.tensor(tire_r_bx1, dtype=torch.float32))  # Slope factor for combined slip Fx reduction
        self.tire_r_bx2 = torch.nn.Parameter(torch.tensor(tire_r_bx2, dtype=torch.float32))  # Variation of slope Fx reduction with kappa
        self.tire_r_cx1 = torch.nn.Parameter(torch.tensor(tire_r_cx1, dtype=torch.float32))  # Shape factor for combined slip Fx reduction
        self.tire_r_ex1 = torch.nn.Parameter(torch.tensor(tire_r_ex1, dtype=torch.float32))  # Curvature factor of combined Fx
        self.tire_r_hx1 = torch.nn.Parameter(torch.tensor(tire_r_hx1, dtype=torch.float32))  # Shift factor for combined slip Fx reduction

        # wheel parameters lateral pure slip
        self.tire_p_cy1 = torch.nn.Parameter(torch.tensor(tire_p_cy1, dtype=torch.float32))  # Shape factor Cfy for lateral forces
        self.tire_p_dy1 = torch.nn.Parameter(torch.tensor(tire_p_dy1, dtype=torch.float32))  # Lateral friction Muy
        self.tire_p_dy3 = torch.nn.Parameter(torch.tensor(tire_p_dy3, dtype=torch.float32))  # Variation of friction Muy with squared camber
        self.tire_p_ey1 = torch.nn.Parameter(torch.tensor(tire_p_ey1, dtype=torch.float32))  # Lateral curvature Efy at Fznom
        self.tire_p_ky1 = torch.nn.Parameter(torch.tensor(tire_p_ky1, dtype=torch.float32))  # Maximum value of stiffness Kfy/Fznom
        self.tire_p_hy1 = torch.nn.Parameter(torch.tensor(tire_p_hy1, dtype=torch.float32))  # Horizontal shift Shy at Fznom
        self.tire_p_hy3 = torch.nn.Parameter(torch.tensor(tire_p_hy3, dtype=torch.float32))  # Variation of shift Shy with camber
        self.tire_p_vy1 = torch.nn.Parameter(torch.tensor(tire_p_vy1, dtype=torch.float32))  # Vertical shift in Svy/Fz at Fznom
        self.tire_p_vy3 = torch.nn.Parameter(torch.tensor(tire_p_vy3, dtype=torch.float32))  # Variation of shift Svy/Fz with camber

        # wheel parameters lateral combined slip
        self.tire_r_by1 = torch.nn.Parameter(torch.tensor(tire_r_by1, dtype=torch.float32))  # Slope factor for combined Fy reduction
        self.tire_r_by2 = torch.nn.Parameter(torch.tensor(tire_r_by2, dtype=torch.float32))  # Variation of slope Fy reduction with alpha
        self.tire_r_by3 = torch.nn.Parameter(torch.tensor(tire_r_by3, dtype=torch.float32))  # Shift term for alpha in slope Fy reduction
        self.tire_r_cy1 = torch.nn.Parameter(torch.tensor(tire_r_cy1, dtype=torch.float32))  # Shape factor for combined Fy reduction
        self.tire_r_ey1 = torch.nn.Parameter(torch.tensor(tire_r_ey1, dtype=torch.float32))  # Curvature factor of combined Fy
        self.tire_r_hy1 = torch.nn.Parameter(torch.tensor(tire_r_hy1, dtype=torch.float32))  # Shift factor for combined Fy reduction
        self.tire_r_vy1 = torch.nn.Parameter(torch.tensor(tire_r_vy1, dtype=torch.float32))  # Kappa induced side force Svyk/Muy*Fz at Fznom
        self.tire_r_vy3 = torch.nn.Parameter(torch.tensor(tire_r_vy3, dtype=torch.float32))  # Variation of Svyk/Muy*Fz with camber
        self.tire_r_vy4 = torch.nn.Parameter(torch.tensor(tire_r_vy4, dtype=torch.float32))  # Variation of Svyk/Muy*Fz with alpha
        self.tire_r_vy5 = torch.nn.Parameter(torch.tensor(tire_r_vy5, dtype=torch.float32))  # Variation of Svyk/Muy*Fz with kappa
        self.tire_r_vy6 = torch.nn.Parameter(torch.tensor(tire_r_vy6, dtype=torch.float32))  # Variation of Svyk/Muy*Fz with atan(kappa)

    def formula_longitudinal(self, kappa, gamma, F_z):

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        # coordinate system transformation
        kappa = -kappa

        S_hx = self.tire_p_hx1
        S_vx = F_z * self.tire_p_vx1

        kappa_x = kappa + S_hx
        mu_x = self.tire_p_dx1 * (1 - self.tire_p_dx3 * gamma ** 2)

        C_x = self.tire_p_cx1
        D_x = mu_x * F_z
        E_x = self.tire_p_ex1
        K_x = F_z * self.tire_p_kx1
        B_x = K_x / (C_x * D_x)

        # magic tire formula
        return D_x * torch.sin(C_x * torch.atan(B_x * kappa_x - E_x * (B_x * kappa_x - torch.atan(B_x * kappa_x))) + S_vx)

    # lateral tire forces
    def formula_lateral(self, alpha, gamma, F_z):

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        # coordinate system transformation
        # alpha = -alpha

        S_hy = torch.sign(gamma) * (self.tire_p_hy1 + self.tire_p_hy3 * torch.abs(gamma))
        S_vy = torch.sign(gamma) * F_z * (self.tire_p_vy1 + self.tire_p_vy3 * torch.abs(gamma))

        alpha_y = alpha + S_hy
        mu_y = self.tire_p_dy1 * (1 - self.tire_p_dy3 * gamma ** 2)

        C_y = self.tire_p_cy1
        D_y = mu_y * F_z
        E_y = self.tire_p_ey1
        K_y = F_z * self.tire_p_ky1  # simplify K_y0 to tire_p_ky1*F_z
        B_y = K_y / (C_y * D_y)

        # magic tire formula
        F_y = D_y * torch.sin(C_y * torch.atan(B_y * alpha_y - E_y * (B_y * alpha_y - torch.atan(B_y * alpha_y)))) + S_vy

        res = []
        res.append(F_y)
        res.append(mu_y)
        return res

    # longitudinal tire forces for combined slip
    def formula_longitudinal_comb(self, kappa, alpha, F0_x):

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        S_hxalpha = self.tire_r_hx1

        alpha_s = alpha + S_hxalpha

        B_xalpha = self.tire_r_bx1 * torch.cos(torch.atan(self.tire_r_bx2 * kappa))
        C_xalpha = self.tire_r_cx1
        E_xalpha = self.tire_r_ex1
        D_xalpha = F0_x / (torch.cos(C_xalpha * torch.atan(
            B_xalpha * S_hxalpha - E_xalpha * (B_xalpha * S_hxalpha - torch.atan(B_xalpha * S_hxalpha)))))

        # magic tire formula
        return D_xalpha * torch.cos(
            C_xalpha * torch.atan(B_xalpha * alpha_s - E_xalpha * (B_xalpha * alpha_s - torch.atan(B_xalpha * alpha_s))))

    # lateral tire forces for combined slip
    def formula_lateral_comb(self, kappa, alpha, gamma, mu_y, F_z, F0_y):

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        S_hykappa = self.tire_r_hy1

        kappa_s = kappa + S_hykappa

        B_ykappa = self.tire_r_by1 * torch.cos(torch.atan(self.tire_r_by2 * (alpha - self.tire_r_by3)))
        C_ykappa = self.tire_r_cy1
        E_ykappa = self.tire_r_ey1
        D_ykappa = F0_y / (torch.cos(C_ykappa * torch.atan(
            B_ykappa * S_hykappa - E_ykappa * (B_ykappa * S_hykappa - torch.atan(B_ykappa * S_hykappa)))))

        D_vykappa = mu_y * F_z * (self.tire_r_vy1 + self.tire_r_vy3 * gamma) * torch.cos(torch.atan(self.tire_r_vy4 * alpha))
        S_vykappa = D_vykappa * torch.sin(self.tire_r_vy5 * torch.atan(self.tire_r_vy6 * kappa))

        # magic tire formula
        return D_ykappa * torch.cos(
            C_ykappa * torch.atan(B_ykappa * kappa_s - E_ykappa * (B_ykappa * kappa_s - torch.atan(B_ykappa * kappa_s)))) + S_vykappa

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
