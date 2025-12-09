from .KoopmanBase import Koopmanlinear, KoopmanBlinear
from .Koopform import Koopformer_PatchTST
from .KANKoopman import KANKoopmanNet
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
    elif args.model == "Koopformer":
        model = Koopformer_PatchTST(
            args.x_dim,
            args.u_dim,
            args.seq_len,
            args.patch_len,
            args.d_model,
        ).to(args.device)
        return model           
    elif args.model == "KANKoopman":
        model = KANKoopmanNet(
            args.x_dim,
            args.u_dim,
            args.kan_layers,
            args.kan_params
        ).to(args.device)
        return model   
    else:
        raise ValueError(f"Model {args.model} not implemented!")