from .KoopmanBase import Koopmanlinear, KoopmanBlinear
from .Koopform import Koopformer,Koopformer_KAN
from .KANKoopman import KANKoopmanNet
from .KoopmanLSTM import KoopmanLSTMlinear,KoopmanLSTMlinear_KAN
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
        model = Koopformer(
            args.x_dim,
            args.u_dim,
            args.seq_len,
            args.d_model,
            args.use_stable,
            args.use_decoder
        ).to(args.device)
        return model     
    elif args.model == "Koopformer_KAN":
        model = Koopformer_KAN(
            args.x_dim,
            args.u_dim,
            args.seq_len,
            args.d_model,
            args.use_stable,
            args.use_decoder
        ).to(args.device)
        return model    
    elif args.model == "KANKoopman":
        model = KANKoopmanNet(
            args.x_dim,
            args.u_dim,
            args.kan_layers,
            args.kan_params,
            args.use_stable,
            args.use_decoder
        ).to(args.device)
        return model   
    elif args.model == "KoopmanLSTMlinear":
        model = KoopmanLSTMlinear(
            args.x_dim,
            args.u_dim,
            args.seq_len,
            args.LSTM_encode_layers,
            args.LSTM_Hidden,
            args.use_stable,
            args.use_decoder
        ).to(args.device)
        return model     
    elif args.model == "KoopmanLSTMlinear_KAN":
        model = KoopmanLSTMlinear_KAN(
            args.x_dim,
            args.u_dim,
            args.seq_len,
            args.kan_layers,
            args.LSTM_Hidden,
            args.use_stable,
            args.use_decoder
        ).to(args.device)
        return model  
    else:
        raise ValueError(f"Model {args.model} not implemented!")