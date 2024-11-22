from . import Dynamics
from ..parameters.definitions import ParameterSample
from .kinematic_bicycle import kinematic_bicycle

import torch
from torch import nn


class SingleTrackDrift(Dynamics, nn.Module):
    """
    This is the Single Track Drift model, from the CommonRoad paper.
    Link: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf
    """

    def __init__(
        self, l, w, m, Iz, lf, lr, hcg, tire_p_dx1, tire_p_dy1, **kwargs
    ) -> None:
        super().__init__()
        self.initial_values = {
            "l": l,
            "w": w,
            "m": m,
            "Iz": Iz,
            "lf": lf,
            "lr": lr,
            "hcg": hcg,
            "h_s": 0.59436,
            "R_w": 0.344,  # wheel radius
            "I_y_w": 1.7*10e3,  # inertia of the wheel
            "T_sb": 0.76,  # torque split between front and rear wheels brake
            "T_se": 1.0,  # torque split between front and rear wheels engine
            # tire parameters from ADAMS handbook
            # longitudinal coefficients
            "tire_p_cx1": 1.6411,  # Shape factor Cfx for longitudinal force
            "tire_p_dx1": tire_p_dx1,  # Longitudinal friction Mux at Fznom
            "tire_p_dx3": 0.0,  # Variation of friction Mux with camber
            "tire_p_ex1": 0.46403,  # Longitudinal curvature Efx at Fznom
            "tire_p_kx1": 22.303,  # Longitudinal slip stiffness Kfx/Fz at Fznom
            "tire_p_hx1": 0.0012297,  # Horizontal shift Shx at Fznom
            "tire_p_vx1": -8.8098e-006,  # Vertical shift Svx/Fz at Fznom
            "tire_r_bx1": 13.276,  # Slope factor for combined slip Fx reduction
            "tire_r_bx2": -13.778,  # Variation of slope Fx reduction with kappa
            "tire_r_cx1": 1.2568,  # Shape factor for combined slip Fx reduction
            "tire_r_ex1": 0.65225,  # Curvature factor of combined Fx
            "tire_r_hx1": 0.0050722,  # Shift factor for combined slip Fx reduction
            # lateral coefficients
            "tire_p_cy1": 1.3507,  # Shape factor Cfy for lateral forces
            "tire_p_dy1": tire_p_dy1,  # Lateral friction Muy
            "tire_p_dy3": -2.8821,  # Variation of friction Muy with squared camber
            "tire_p_ey1": -0.0074722,  # Lateral curvature Efy at Fznom
            "tire_p_ky1": -21.92,  # Maximum value of stiffness Kfy/Fznom
            "tire_p_hy1": 0.0026747,  # Horizontal shift Shy at Fznom
            "tire_p_hy3": 0.031415,  # Variation of shift Shy with camber
            "tire_p_vy1": 0.037318,  # Vertical shift in Svy/Fz at Fznom
            "tire_p_vy3": -0.32931,  # Variation of shift Svy/Fz with camber
            "tire_r_by1": 7.1433,  # Slope factor for combined Fy reduction
            "tire_r_by2": 9.1916,  # Variation of slope Fy reduction with alpha
            "tire_r_by3": -0.027856,  # Shift term for alpha in slope Fy reduction
            "tire_r_cy1": 1.0719,  # Shape factor for combined Fy reduction
            "tire_r_ey1": -0.27572,  # Curvature factor of combined Fy
            "tire_r_hy1": 5.7448e-006,  # Shift factor for combined Fy reduction
            "tire_r_vy1": -0.027825,  # Kappa induced side force Svyk/Muy*Fz at Fznom
            "tire_r_vy3": -0.27568,  # Variation of Svyk/Muy*Fz with camber
            "tire_r_vy4": 12.12,  # Variation of Svyk/Muy*Fz with alpha
            "tire_r_vy5": 1.9,  # Variation of Svyk/Muy*Fz with kappa
            "tire_r_vy6": -10.704,  # Variation of Svyk/Muy*Fz with atan(kappa)
        }
        self.parameter_list = list(self.initial_values.keys())

        self.g = 9.81

    def formula_longitudinal(self, kappa, gamma, F_z, params: dict):
        # longitudinal coefficients
        tire_p_cx1 = params["tire_p_cx1"]  # Shape factor Cfx for longitudinal force
        tire_p_dx1 = params["tire_p_dx1"]  # Longitudinal friction Mux at Fznom
        tire_p_dx3 = params["tire_p_dx3"]  # Variation of friction Mux with camber
        tire_p_ex1 = params["tire_p_ex1"]  # Longitudinal curvature Efx at Fznom
        tire_p_kx1 = params["tire_p_kx1"]  # Longitudinal slip stiffness Kfx/Fz at Fznom
        tire_p_hx1 = params["tire_p_hx1"]  # Horizontal shift Shx at Fznom
        tire_p_vx1 = params["tire_p_vx1"]  # Vertical shift Svx/Fz at Fznom

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        # coordinate system transformation
        kappa = -kappa

        S_hx = tire_p_hx1
        S_vx = F_z * tire_p_vx1

        kappa_x = kappa + S_hx
        mu_x = tire_p_dx1 * (1 - tire_p_dx3 * gamma**2)

        C_x = tire_p_cx1
        D_x = mu_x * F_z
        E_x = tire_p_ex1
        K_x = F_z * tire_p_kx1
        B_x = K_x / (C_x * D_x)

        # magic tire formula
        return D_x * torch.sin(
            C_x
            * torch.arctan(
                B_x * kappa_x - E_x * (B_x * kappa_x - torch.arctan(B_x * kappa_x))
            )
            + S_vx
        )

    def formula_lateral(self, alpha, gamma, F_z, params: dict):
        # lateral coefficients
        tire_p_cy1 = params["tire_p_cy1"]  # Shape factor Cfy for lateral forces
        tire_p_dy1 = params["tire_p_dy1"]  # Lateral friction Muy
        tire_p_dy3 = params[
            "tire_p_dy3"
        ]  # Variation of friction Muy with squared camber
        tire_p_ey1 = params["tire_p_ey1"]  # Lateral curvature Efy at Fznom
        tire_p_ky1 = params["tire_p_ky1"]  # Maximum value of stiffness Kfy/Fznom
        tire_p_hy1 = params["tire_p_hy1"]  # Horizontal shift Shy at Fznom
        tire_p_hy3 = params["tire_p_hy3"]  # Variation of shift Shy with camber
        tire_p_vy1 = params["tire_p_vy1"]  # Vertical shift in Svy/Fz at Fznom
        tire_p_vy3 = params["tire_p_vy3"]  # Variation of shift Svy/Fz with camber

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        # coordinate system transformation
        # alpha = -alpha

        S_hy = torch.sign(gamma) * (tire_p_hy1 + tire_p_hy3 * torch.abs(gamma))
        S_vy = torch.sign(gamma) * F_z * (tire_p_vy1 + tire_p_vy3 * torch.abs(gamma))

        alpha_y = alpha + S_hy
        mu_y = tire_p_dy1 * (1 - tire_p_dy3 * gamma**2)

        C_y = tire_p_cy1
        D_y = mu_y * F_z
        E_y = tire_p_ey1
        K_y = F_z * tire_p_ky1  # simplify K_y0 to tire_p_ky1*F_z
        B_y = K_y / (C_y * D_y)

        # magic tire formula
        F_y = (
            D_y
            * torch.sin(
                C_y
                * torch.arctan(
                    B_y * alpha_y - E_y * (B_y * alpha_y - torch.arctan(B_y * alpha_y))
                )
            )
            + S_vy
        )
        return F_y, mu_y

    def formula_longitudinal_comb(self, kappa, alpha, F0_x, params: dict):
        # longitudinal coefficients
        tire_r_bx1 = params["tire_r_bx1"]  # Slope factor for combined slip Fx reduction
        tire_r_bx2 = params["tire_r_bx2"]  # Variation of slope Fx reduction with kappa
        tire_r_cx1 = params["tire_r_cx1"]  # Shape factor for combined slip Fx reduction
        tire_r_ex1 = params["tire_r_ex1"]  # Curvature factor of combined Fx
        tire_r_hx1 = params["tire_r_hx1"]  # Shift factor for combined slip Fx reduction

        # turn slip '' neglected, so xi_i=1
        # all scaling factors lambda = 1

        S_hxalpha = tire_r_hx1

        alpha_s = alpha + S_hxalpha

        B_xalpha = tire_r_bx1 * torch.cos(torch.arctan(tire_r_bx2 * kappa))
        C_xalpha = tire_r_cx1
        E_xalpha = tire_r_ex1
        D_xalpha = F0_x / (
            torch.cos(
                C_xalpha
                * torch.arctan(
                    B_xalpha * S_hxalpha
                    - E_xalpha
                    * (B_xalpha * S_hxalpha - torch.arctan(B_xalpha * S_hxalpha))
                )
            )
        )

        # magic tire formula
        return D_xalpha * torch.cos(
            C_xalpha
            * torch.arctan(
                B_xalpha * alpha_s
                - E_xalpha * (B_xalpha * alpha_s - torch.arctan(B_xalpha * alpha_s))
            )
        )

    def formula_lateral_comb(self, kappa, alpha, gamma, mu_y, F_z, F0_y, params: dict):
        # lateral coefficients
        tire_r_by1 = params["tire_r_by1"]  # Slope factor for combined Fy reduction
        tire_r_by2 = params["tire_r_by2"]  # Variation of slope Fy reduction with alpha
        tire_r_by3 = params["tire_r_by3"]  # Shift term for alpha in slope Fy reduction
        tire_r_cy1 = params["tire_r_cy1"]  # Shape factor for combined Fy reduction
        tire_r_ey1 = params["tire_r_ey1"]  # Curvature factor of combined Fy
        tire_r_hy1 = params["tire_r_hy1"]  # Shift factor for combined Fy reduction
        tire_r_vy1 = params[
            "tire_r_vy1"
        ]  # Kappa induced side force Svyk/Muy*Fz at Fznom
        tire_r_vy3 = params["tire_r_vy3"]  # Variation of Svyk/Muy*Fz with camber
        tire_r_vy4 = params["tire_r_vy4"]  # Variation of Svyk/Muy*Fz with alpha
        tire_r_vy5 = params["tire_r_vy5"]  # Variation of Svyk/Muy*Fz with kappa
        tire_r_vy6 = params["tire_r_vy6"]  # Variation of Svyk/Muy*Fz with atan(kappa)

        # turn slip is neglected, so xi_i=1
        # all scaling factors lambda = 1

        S_hykappa = tire_r_hy1

        kappa_s = kappa + S_hykappa

        B_ykappa = tire_r_by1 * torch.cos(
            torch.arctan(tire_r_by2 * (alpha - tire_r_by3))
        )
        C_ykappa = tire_r_cy1
        E_ykappa = tire_r_ey1
        D_ykappa = F0_y / (
            torch.cos(
                C_ykappa
                * torch.arctan(
                    B_ykappa * S_hykappa
                    - E_ykappa
                    * (B_ykappa * S_hykappa - torch.arctan(B_ykappa * S_hykappa))
                )
            )
        )

        D_vykappa = (
            mu_y
            * F_z
            * (tire_r_vy1 + tire_r_vy3 * gamma)
            * torch.cos(torch.arctan(tire_r_vy4 * alpha))
        )
        S_vykappa = D_vykappa * torch.sin(tire_r_vy5 * torch.arctan(tire_r_vy6 * kappa))

        # magic tire formula
        return (
            D_ykappa
            * torch.cos(
                C_ykappa
                * torch.arctan(
                    B_ykappa * kappa_s
                    - E_ykappa * (B_ykappa * kappa_s - torch.arctan(B_ykappa * kappa_s))
                )
            )
            + S_vykappa
        )

    def forward(self, states, control_inputs, params: ParameterSample):
        """Get the evaluated ODEs of the state at this point

        Args:
            states (): Shape of (B, 8) or (8)
                [X, Y, V, YAW, YAW_RATE, SLIP_ANGLE]
            control_inputs (): Shape of (B, 2) or (2)
                [STEER_ANGLE, ACCEL]
        """
        # x0 = x-position in a global coordinate system
        # x1 = y-position in a global coordinate system
        # x2 = steering angle of front wheels
        # x3 = velocity at vehicle center
        # x4 = yaw angle
        # x5 = yaw rate
        # x6 = slip angle at vehicle center
        # x7 = front wheel angular speed
        # x8 = rear wheel angular speed

        # u0 = steering angle velocity of front wheels
        # u1 = longitudinal acceleration

        batch_mode = True if len(states.shape) == 2 else False
        X, Y, STEER_ANGLE, V, YAW, YAW_RATE, SLIP_ANGLE, OMEGA_F, OMEGA_R = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        )
        STEER_VEL, ACCEL = 0, 1

        # states[..., OMEGA_F] = states[..., V]*torch.cos(states[..., SLIP_ANGLE])*torch.cos(states[..., STEER_ANGLE])/params['R_w']
        # states[..., OMEGA_R] = states[..., V]*torch.cos(states[..., SLIP_ANGLE])/params['R_w']

        v_s = 0.2
        v_b = 0.05
        v_min = v_s / 2

        # compute lateral tire slip angles
        alpha_f = (
            torch.atan(
                (
                    states[..., V] * torch.sin(states[..., SLIP_ANGLE])
                    + states[..., YAW_RATE] * params["lf"]
                )
                / (states[..., V] * torch.cos(states[..., SLIP_ANGLE]))
            )
            - states[..., STEER_ANGLE]
        )
        mask = states[..., V] > v_min
        alpha_f = torch.where(mask, alpha_f, torch.zeros_like(alpha_f))

        alpha_r = torch.atan(
            (
                states[..., V] * torch.sin(states[..., SLIP_ANGLE])
                - states[..., YAW_RATE] * params["lr"]
            )
            / (states[..., V] * torch.cos(states[..., SLIP_ANGLE]))
        )
        mask = states[..., V] > v_min
        alpha_r = torch.where(mask, alpha_r, torch.zeros_like(alpha_r))

        # compute vertical tire forces
        F_zf = (
            params["m"]
            * (-control_inputs[..., ACCEL] * params["h_s"] + self.g * params["lr"])
            / (params["lr"] + params["lf"])
        )
        F_zr = (
            params["m"]
            * (control_inputs[..., ACCEL] * params["h_s"] + self.g * params["lf"])
            / (params["lr"] + params["lf"])
        )

        # compute front and rear tire speeds
        u_wf = states[..., V] * torch.cos(states[..., SLIP_ANGLE]) * torch.cos(
            states[..., STEER_ANGLE]
        ) + (
            states[..., V] * torch.sin(states[..., SLIP_ANGLE])
            + params["lf"] * states[..., YAW_RATE]
        ) * torch.sin(states[..., STEER_ANGLE])
        neg_u_wf = u_wf < 0
        u_wf = torch.where(neg_u_wf, torch.zeros_like(u_wf), u_wf)

        u_wr = states[..., V] * torch.cos(states[..., SLIP_ANGLE])
        neg_u_wr = u_wr < 0
        u_wr = torch.where(neg_u_wr, torch.zeros_like(u_wr), u_wr)

        # compute longitudinal tire slip
        s_f = 1 - params["R_w"] * states[..., OMEGA_F] / torch.maximum(
            u_wf, v_min * torch.ones_like(u_wf)
        )
        s_r = 1 - params["R_w"] * states[..., OMEGA_R] / torch.maximum(
            u_wr, v_min * torch.ones_like(u_wf)
        )

        # compute tire forces (Pacejka)
        zeros = torch.zeros_like(states[..., V])
        # pure slip longitudinal forces
        F0_xf = self.formula_longitudinal(s_f, zeros, F_zf, params)
        F0_xr = self.formula_longitudinal(s_r, zeros, F_zr, params)

        # pure slip lateral forces
        res = self.formula_lateral(alpha_f, zeros, F_zf, params)
        F0_yf = res[0]
        mu_yf = res[1]
        res = self.formula_lateral(alpha_r, zeros, F_zr, params)
        F0_yr = res[0]
        mu_yr = res[1]

        # combined slip longitudinal forces
        F_xf = self.formula_longitudinal_comb(s_f, alpha_f, F0_xf, params)
        F_xr = self.formula_longitudinal_comb(s_r, alpha_r, F0_xr, params)

        # combined slip lateral forces
        F_yf = self.formula_lateral_comb(s_f, alpha_f, 0, mu_yf, F_zf, F0_yf, params)
        F_yr = self.formula_lateral_comb(s_r, alpha_r, 0, mu_yr, F_zr, F0_yr, params)

        # convert acceleration input to brake and engine torque
        T = params["m"] * params["R_w"] * control_inputs[..., ACCEL]
        T_B = torch.where(control_inputs[..., ACCEL] > 0, torch.zeros_like(T), T)
        T_E = torch.where(control_inputs[..., ACCEL] > 0, T, torch.zeros_like(T))

        d_v = (
            1
            / params["m"]
            * (
                -F_yf * torch.sin(states[..., STEER_ANGLE] - states[..., SLIP_ANGLE])
                + F_yr * torch.sin(states[..., SLIP_ANGLE])
                + F_xr * torch.cos(states[..., SLIP_ANGLE])
                + F_xf * torch.cos(states[..., STEER_ANGLE] - states[..., SLIP_ANGLE])
            )
        )
        dd_psi = (
            1
            / params["Iz"]
            * (
                F_yf * torch.cos(states[..., STEER_ANGLE]) * params["lf"]
                - F_yr * params["lr"]
                + F_xf * torch.sin(states[..., STEER_ANGLE]) * params["lf"]
            )
        )
        d_beta = -states[..., YAW_RATE] + 1 / (
            params["m"] * states[..., V]
        ) * (
            F_yf * torch.cos(states[..., STEER_ANGLE] - states[..., SLIP_ANGLE])
            + F_yr * torch.cos(states[..., SLIP_ANGLE])
            - F_xr * torch.sin(states[..., SLIP_ANGLE])
            + F_xf * torch.sin(states[..., STEER_ANGLE] - states[..., SLIP_ANGLE])
        )
        mask = states[..., V] > v_min
        d_beta = torch.where(mask, d_beta, zeros)

        d_omega_f = (
            1
            / params["I_y_w"]
            * (-params["R_w"] * F_xf + params["T_sb"] * T_B + params["T_se"] * T_E)
        )
        mask = states[..., OMEGA_F] >= 0
        d_omega_f = torch.where(mask, d_omega_f, zeros)

        d_omega_r = (
            1
            / params["I_y_w"]
            * (
                -params["R_w"] * F_xr
                + (1 - params["T_sb"]) * T_B
                + (1 - params["T_se"]) * T_E
            )
        )
        mask = states[..., OMEGA_R] >= 0
        d_omega_r = torch.where(mask, d_omega_r, zeros)

        state_ks = torch.stack(
            [
                states[..., X],
                states[..., Y],
                states[..., STEER_ANGLE],
                states[..., V],
                states[..., YAW],
            ],
        )

        diff_ks = kinematic_bicycle(state_ks, control_inputs, params)

        d_beta_ks = (params["lr"] * control_inputs[..., STEER_VEL]) / (
            (params["lf"] + params["lr"])
            * torch.cos(states[..., STEER_ANGLE]) ** 2
            * (
                1
                + (
                    torch.tan(states[..., STEER_ANGLE]) ** 2
                    * params["lr"]
                    / (params["lf"] + params["lr"])
                )
                ** 2
            )
        )

        dd_psi_ks = (
            1
            / (params["lr"] + params["lf"])
            * (
                control_inputs[..., 1]
                * torch.cos(states[..., 6])
                * torch.tan(states[..., 2])
                - states[..., 3]
                * torch.sin(states[..., 6])
                * d_beta_ks
                * torch.tan(states[..., 2])
                + states[..., 3]
                * torch.cos(states[..., 6])
                * control_inputs[..., 0]
                / torch.cos(states[..., 2]) ** 2
            )
        )

        d_omega_f_ks = (1 / 0.02) * (u_wf / params['R_w'] - states[..., OMEGA_F])
        d_omega_r_ks = (1 / 0.02) * (u_wr / params['R_w'] - states[..., OMEGA_R])

        w_std = 0.5 * (torch.tanh((states[..., V] - v_s)/v_b) + 1)
        w_ks = 1 - w_std

        diff = torch.zeros_like(states)
        # update states
        diff[..., X] = states[..., V] * torch.cos(
            states[..., YAW] + states[..., SLIP_ANGLE]
        )
        diff[..., Y] = states[..., V] * torch.sin(
            states[..., YAW] + states[..., SLIP_ANGLE]
        )
        diff[..., STEER_ANGLE] = control_inputs[..., STEER_VEL]
        diff[..., V] = w_std * d_v + w_ks * diff_ks[..., 3]
        diff[..., YAW] = w_std * states[..., YAW_RATE] + w_ks * diff_ks[..., 4]
        diff[..., YAW_RATE] = w_std * dd_psi + w_ks * dd_psi_ks
        diff[..., SLIP_ANGLE] = w_std * d_beta + w_ks * d_beta_ks
        diff[..., OMEGA_F] = w_std * d_omega_f + w_ks * d_omega_f_ks
        diff[..., OMEGA_R] = w_std * d_omega_r + w_ks * d_omega_r_ks

        return diff
