import torch
from physicsnemo.sym.hydra import instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.deploy.onnx import export_to_onnx_stream, run_onnx_inference
import physicsnemo.sym

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Recreate the model using the same config and keys as in training
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    # Load trained weights
    flow_net.load_state_dict(torch.load("/home/genai/outputs/bb/flow_network.0.pth"))
    flow_net.eval()

    # Dummy input matching the trained model's input: [batch_size, 2]
    dummy_input = torch.randn(4, 2)

    # Export ONNX (only export the inner PyTorch model)
    onnx_stream = export_to_onnx_stream(flow_net._impl, dummy_input, verbose=True)

    # Save ONNX model
    with open("flow_network.onnx", "wb") as f:
        f.write(onnx_stream)

    # Optional: Verify ONNX output against PyTorch
    torch_output = flow_net._impl(dummy_input)
    onnx_output = run_onnx_inference(onnx_stream, dummy_input)
    onnx_output = torch.Tensor(onnx_output[0])

    assert torch.allclose(torch_output, onnx_output, atol=1e-4)
    print("✅ ONNX export successful and outputs match.")

if __name__ == "__main__":
    run()

