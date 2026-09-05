"""Uncompiled PyTorch loaders for the separate eager MD runners."""


def assert_eager_module(model):
    """Reject TorchScript and torch.compile modules, including nested modules."""
    import torch
    from torch._dynamo.eval_frame import OptimizedModule

    for name, module in model.named_modules():
        if isinstance(module, (torch.jit.ScriptModule, OptimizedModule)):
            raise RuntimeError(f"Compiled module in eager model: {name or '<root>'}")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise RuntimeError(f"Compiled call in eager model: {name or '<root>'}")


def load_eager_pet(*, model, size, version, checkpoint_path=None):
    """Follow UPET 0.2.6's loader but omit its final torch.jit.script call."""
    from upet._models import _get_upet_exported_atomistic_model
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    if checkpoint_path is None:
        filename = f"models/{model}-{size}-v{version}.ckpt"
        cached = try_to_load_from_cache("lab-cosmo/upet", filename)
        checkpoint_path = cached if isinstance(cached, str) else hf_hub_download(
            "lab-cosmo/upet", filename=filename
        )
    model = _get_upet_exported_atomistic_model(
        model=model, size=size, version=version, checkpoint_path=checkpoint_path
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert_eager_module(model)
    return model
