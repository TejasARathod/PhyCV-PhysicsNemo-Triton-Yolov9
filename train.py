import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.interpolate
from sympy import Symbol, Eq
import pandas as pd
import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.utils.io import InferencerPlotter


class CustomInferencerPlotter(InferencerPlotter):
    def __call__(self, invar, pred_outvar):
        x, y = invar["x"][:, 0], invar["y"][:, 0]
        extent = (x.min(), x.max(), y.min(), y.max())

        u_pred = pred_outvar["u"][:, 0]
        v_pred = pred_outvar["v"][:, 0]
        p_pred = pred_outvar["p"][:, 0]

        u_pred_np, v_pred_np, p_pred_np = self.interpolate_output(x, y, [u_pred, v_pred, p_pred], extent)

        f, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
        f.suptitle("Predicted Fields")

        axs[0].imshow(u_pred_np.T, origin="lower", extent=extent)
        axs[0].set_title("Predicted u")
        axs[1].imshow(v_pred_np.T, origin="lower", extent=extent)
        axs[1].set_title("Predicted v")
        axs[2].imshow(p_pred_np.T, origin="lower", extent=extent)
        axs[2].set_title("Predicted p")

        plt.tight_layout()
        return [(f, "pred_fields")]

    @staticmethod
    def interpolate_output(x, y, us, extent):
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100),
            np.linspace(extent[2], extent[3], 100),
            indexing="ij",
        )
        us_interp = [
            scipy.interpolate.griddata((x, y), u, tuple(xyi), method="linear") for u in us
        ]
        return us_interp


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Define PDE and architecture without time
    ns = NavierStokes(nu=1.5e-5, rho=1.225, dim=2, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # Geometry domain (1m x 1m)
    width, height = 1.0, 1.0
    rec = Rectangle((0.0, 0.0), (width, height))

    # Create domain
    domain = Domain()

    x, y = Symbol("x"), Symbol("y")

    # No-slip condition on left, right, bottom boundaries
    for condition, eq in zip(["left", "right", "bottom"], [Eq(x, 0.0), Eq(x, 1.0), Eq(y, 0.0)]):
        domain.add_constraint(
            PointwiseBoundaryConstraint(
                nodes=nodes,
                geometry=rec,
                outvar={"u": 0.0, "v": 0.0},
                batch_size=cfg.batch_size.NoSlip,
                criteria=eq,
            ),
            name=condition
        )

    # Interior: Navier-Stokes constraints
    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=rec,
            outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            batch_size=cfg.batch_size.Interior,
        ),
        name="interior",
    )

    # Global monitor for mass and momentum imbalance
    global_monitor = PointwiseMonitor(
        rec.sample_interior(200, bounds={x: (0.0, 1.0), y: (0.0, 1.0)}),
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "mass_imbalance": lambda var: torch.sum(var["area"] * torch.abs(var["continuity"])),
            "momentum_imbalance": lambda var: torch.sum(
                var["area"] * (torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"]))
            ),
        },
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(global_monitor)

    # Load trajectory data but ignore time
    traj_df = pd.read_csv("/home/genai/normalized_ball_trajectory.csv")

    # Prepare input for inference (no time)
    invar = {
        "x": traj_df["x_norm"].to_numpy().reshape(-1, 1).astype(np.float32),
        "y": traj_df["y_norm"].to_numpy().reshape(-1, 1).astype(np.float32),
    }

    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=invar,
        output_names=["u", "v", "p"],
        batch_size=32,
        plotter=CustomInferencerPlotter(),
    )
    domain.add_inferencer(inferencer, "basketball_flow")

    # Solve
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()

