import torch
import numpy as np
from physicsnemo.sym.hydra import instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.hydra import PhysicsNeMoConfig
import physicsnemo.sym


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Instantiate the architecture
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    # Load trained weights
    flow_net.load_state_dict(torch.load("/home/genai/outputs/bb/flow_network.0.pth"))
    flow_net.eval()

    # Generate 4 random x, y values ∈ [0, 1)
    x_vals = torch.rand(4, 1)
    y_vals = torch.rand(4, 1)

    # Prepare input dict using plain string keys
    input_dict = {
        "x": x_vals,
        "y": y_vals,
    }

    # Run inference
    with torch.no_grad():
        output = flow_net(input_dict)

    # Print results
    for i in range(4):
        x = x_vals[i].item()
        y = y_vals[i].item()
        u = output["u"][i].item()
        v = output["v"][i].item()
        p = output["p"][i].item()
        print(f"Input {i+1}: x = {x:.4f}, y = {y:.4f} → u = {u:.4f}, v = {v:.4f}, p = {p:.4f}")


if __name__ == "__main__":
    run()

