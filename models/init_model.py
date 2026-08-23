"""Factory for the four core Koopman baselines retained in the clean branch."""

from .InvertKoopman import InvertKoopmanNetBLinear, InvertKoopmanNetLinear
from .KoopmanBase import KoopmanBlinear, Koopmanlinear


def init_model(args):
    if args.x_dim != 12 or args.u_dim != 6:
        raise ValueError("UR5e torque models require x_dim=12 and u_dim=6")
    if args.model == "DKUC":
        return Koopmanlinear(args.x_dim, args.u_dim, args.layers, args.use_stable).to(args.device)
    if args.model == "DBKN":
        return KoopmanBlinear(args.x_dim, args.u_dim, args.layers, args.u_z, args.use_stable).to(args.device)
    if args.model == "IKN":
        return InvertKoopmanNetLinear(
            args.x_dim,
            args.x_blocks,
            args.x_channels,
            args.x_hiddens,
            args.u_dim,
            args.u_blocks,
            args.u_channels,
            args.u_hiddens,
            args.use_stable,
        ).to(args.device)
    if args.model == "IBKN":
        return InvertKoopmanNetBLinear(
            args.x_dim,
            args.x_blocks,
            args.x_channels,
            args.x_hiddens,
            args.u_dim,
            args.u_blocks,
            args.u_channels,
            args.u_hiddens,
            args.u_z,
            args.use_stable,
        ).to(args.device)
    raise ValueError(f"Model {args.model!r} is not a single model")
