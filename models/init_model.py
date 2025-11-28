from .KoopmanBase import Koopmanlinear, KoopmanBlinear
def init_model(args):
    print(f"Initiating {args.model}")
    if args.model == "DKUC":
        model = Koopmanlinear(
            args.x_dim,
            args.u_dim,
            args.layers
        ).to(args.device)
        return model 
    elif args.model == "DBKN":
        model = KoopmanBlinear(
            args.x_dim,
            args.u_dim,
            args.layers,
            args.u_z
        ).to(args.device)
        return model
    raise ValueError(f"Model {args.model} not implemented!")