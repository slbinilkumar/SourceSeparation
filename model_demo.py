from source_separation.data_objects import MidiDataset, get_instrument_id
from source_separation.hparams import hparams
from source_separation.parser import parse_args
from source_separation.model import Model
from pathlib import Path
import torch


if __name__ == "__main__":
    args = parse_args()
    source_instruments = args.source_instruments
    target_instruments = args.target_instruments
    get_instruments_id = lambda l: list(map(get_instrument_id, l.split(",")))
    
    dataset = MidiDataset(
        root=args.dataset_root,
        is_train=True,
        hparams=hparams,
    )
    
    dataloader = dataset.generate(
        source_instruments=get_instruments_id(source_instruments),
        target_instruments=get_instruments_id(target_instruments),
        batch_size=args.batch_size,
        n_threads=4,
        music_buffer_size=8,    # Careful, high values can have a high RAM impact
        quickstart=True,        # For quick debugging
    )

    # # If you want to have a look at the data
    # import sounddevice as sd
    # x_train, y_train = next(dataloader)
    # for i in range(args.batch_size):
    #     print("Playing chunk %d" % i)
    #     sd.play(x_train[i], 44100, blocking=True)
    # quit()

    # Create the model and the optimizer
    model = Model().cuda()
    learning_rate_init = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1
    save_every = 100

    # Load any existing model
    state_fpath = Path("model.pt")
    if state_fpath.exists():
        print("Found existing model \"%s\", loading it and resuming training." % state_fpath)
        checkpoint = torch.load(state_fpath)
        init_step = checkpoint["step"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer.param_groups[0]["lr"] = learning_rate_init
    else:
        print("No model \"%s\" found, starting training from scratch." % state_fpath)
        
    # Set the model to training mode
    model.train()

    # Training loop
    for step, batch in enumerate(dataloader, init_step):
        # Forward pass
        x, y_true = torch.from_numpy(batch).cuda()
        y_pred = model(x)
        loss = model.loss(y_pred, y_true)
        print(loss.item)
    
        # Backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()
    
        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
