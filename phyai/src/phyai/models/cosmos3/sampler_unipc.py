"""UniPC multistep sampler for Cosmos3 flow-matching sampling.

https://arxiv.org/abs/2302.04867

TODO(wch): make this scheduler public for other models.
"""

from __future__ import annotations

import numpy as np
import torch


class UniPCMultistepSampler:
    """Order-2 UniPC (bh2) for flow-prediction / predict-x0 sampling."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        sigma_min: float = 0.147,
        sigma_max: float = 200.0,
        karras_rho: float = 7.0,
        flow_shift: float = 1.0,
        lower_order_final: bool = True,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.solver_order = int(solver_order)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.karras_rho = float(karras_rho)
        self.flow_shift = float(flow_shift)
        self.lower_order_final = bool(lower_order_final)
        self.solver_type = "bh2"
        self.predict_x0 = True

        # Per-run state (populated by set_timesteps).
        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.model_outputs: list[torch.Tensor | None] = [None] * self.solver_order
        self.timestep_list: list[torch.Tensor | None] = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        self.this_order = 0
        self._step_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(
        self, num_inference_steps: int, device: torch.device | str | None = None
    ) -> None:
        """Build the Karras→flow-remapped sigma schedule (+ trailing 0)."""
        n = int(num_inference_steps)
        # Karras sigmas from the configured bounds (alphas_cumprod not needed on
        # the flow path: _convert_to_karras reads sigma_min/max directly).
        ramp = np.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1.0 / self.karras_rho)
        max_inv_rho = self.sigma_max ** (1.0 / self.karras_rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.karras_rho
        # Flow remap (matches diffusers use_flow_sigmas + use_karras_sigmas path):
        # sigmas/(sigmas+1), then timesteps = sigma * num_train_timesteps. flow_shift
        # is not applied on the karras path (the diffusers __call__ leaves it at the
        # scheduler_config 1.0), so it is intentionally unused beyond record-keeping.
        sigmas = sigmas / (sigmas + 1.0)
        timesteps = (sigmas * self.num_train_timesteps).copy()
        # final_sigmas_type="zero": append a trailing zero sigma.
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(
            device=device, dtype=torch.int64
        )
        self.num_inference_steps = n
        # reset solver state
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 0
        self._step_index = None

    @staticmethod
    def _alpha_sigma(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # flow sigmas: alpha_t = 1 - sigma, sigma_t = sigma.
        return 1.0 - sigma, sigma

    def _sigma(self, idx: int) -> torch.Tensor:
        return self.sigmas[idx]

    def convert_model_output(
        self, model_output: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        sigma_t = self.sigmas[self._step_index].to(sample.device, sample.dtype)
        return sample - sigma_t * model_output  # x0 prediction

    def _p_update(self, sample: torch.Tensor, order: int) -> torch.Tensor:
        m0 = self.model_outputs[-1]
        x = sample
        device, dtype = x.device, x.dtype
        si = self._step_index
        sigma_t = self.sigmas[si + 1].to(device)
        sigma_s0 = self.sigmas[si].to(device)
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks: list = []
        d1s: list = []
        for i in range(1, order):
            mi = self.model_outputs[-(i + 1)]
            a_si, s_si = self._alpha_sigma(self.sigmas[si - i].to(device))
            lambda_si = torch.log(a_si) - torch.log(s_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)
        rks.append(torch.tensor(1.0, device=device))
        rks = torch.stack(rks).to(device)

        hh = -h  # predict_x0
        h_phi_1 = torch.expm1(hh)
        b_h = torch.expm1(hh)  # bh2

        if d1s:
            d1s_t = torch.stack(d1s, dim=1)
            # order==2 shortcut (no linear solve).
            rhos_p = torch.tensor([0.5], dtype=dtype, device=device)
        else:
            d1s_t = None

        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if d1s_t is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s_t)
            x_t = x_t - alpha_t * b_h * pred_res
        return x_t.to(dtype)

    def _c_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        m0 = self.model_outputs[-1]
        x = last_sample
        device, dtype = this_sample.device, this_sample.dtype
        si = self._step_index
        sigma_t = self.sigmas[si].to(device)
        sigma_s0 = self.sigmas[si - 1].to(device)
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks: list = []
        d1s: list = []
        for i in range(1, order):
            mi = self.model_outputs[-(i + 1)]
            a_si, s_si = self._alpha_sigma(self.sigmas[si - (i + 1)].to(device))
            lambda_si = torch.log(a_si) - torch.log(s_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)
        rks.append(torch.tensor(1.0, device=device))
        rks = torch.stack(rks).to(device)

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        b_h = torch.expm1(hh)  # bh2

        r_rows: list = []
        b_rows: list = []
        factorial_i = 1
        for i in range(1, order + 1):
            r_rows.append(torch.pow(rks, i - 1))
            b_rows.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        r_mat = torch.stack(r_rows)
        b_vec = torch.stack(b_rows).to(device)

        d1s_t = torch.stack(d1s, dim=1) if d1s else None
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(r_mat, b_vec).to(device=device, dtype=dtype)

        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        corr_res = (
            torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s_t)
            if d1s_t is not None
            else 0.0
        )
        d1_t = this_model_output - m0
        x_t = x_t - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return x_t.to(dtype)

    def step(
        self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        """One UniPC step. Returns the updated sample (no dict wrapper)."""
        if self.num_inference_steps is None:
            raise RuntimeError("call set_timesteps() before step().")
        if self._step_index is None:
            self._step_index = 0  # loop always starts at the first timestep

        use_corrector = self._step_index > 0 and self.last_sample is not None
        m_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self._c_update(
                this_model_output=m_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # shift history (solver_order - 1 slots)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = m_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self._step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)

        self.last_sample = sample
        prev_sample = self._p_update(sample=sample, order=self.this_order)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1
        return prev_sample


__all__ = ["UniPCMultistepSampler"]
