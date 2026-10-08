import torch

def resolve_device(device: torch.device | str) -> torch.device:
    """Resolve an implicit CUDA device to its active logical index."""
    device = torch.device(device)

    if device.type == "cuda" and device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Cannot resolve implicit CUDA device: CUDA unavailable"
            )
        return torch.device("cuda", torch.cuda.current_device())

    return device



def same_device(
    lhs: torch.device | str,
    rhs: torch.device | str,
) -> bool:
    """Compare actual logical devices without moving tensors."""
    return resolve_device(lhs) == resolve_device(rhs)