from source_separation.visualizations import Visualizations
from source_separation.data_objects import MidiDataset
from source_separation.model import SimpleConvolutionalModel, spectrogram_loss
from pathlib import Path
import numpy as np
import torch


def train(args, hparams):
    # Initialize visdom
    vis = Visualizations(args.run_name, averaging_window=25, auto_open_browser=True)
    
    # Create the dataset
    dataset = MidiDataset(
        root=args.dataset_root,
        is_train=True,
        hparams=hparams,
    )
    data_iterator = dataset.generate(
        source_instruments=args.source_instruments,
        target_instruments=args.target_instruments,
        batch_size=args.batch_size,
        n_threads=4,
        chunk_reuse=args.chunk_reuse,
        chunk_pool_size=args.chunk_pool_size,
        quickstart=args.quickstart,
    )

    # Create the model and the optimizer
    model = SimpleConvolutionalModel(hparams).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate_init)
    init_step = 1

    # Load any existing model
    state_fpath = Path("%s.pt" % args.run_name)
    if state_fpath.exists():
        print("Found existing model \"%s\", loading it and resuming training." % state_fpath)
        checkpoint = torch.load(state_fpath)
        init_step = checkpoint["step"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer.param_groups[0]["lr"] = hparams.learning_rate_init
    else:
        print("No model \"%s\" found, starting training from scratch." % state_fpath)
        
    # Set the model to training mode
    model.train()
    
    # Setup the visualizations environment
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_params(args.__dict__, "Arguments")
    vis.log_params(hparams.__dict__, "Hyperparameters")
    vis.log_implementation({"Device": device_name})

    # Training loop
    loss_buffer = []
    for step, batch in enumerate(data_iterator, init_step):
        # Forward pass
        x, y_true = torch.from_numpy(batch).cuda()
        y_pred = model(x)
        loss = spectrogram_loss(y_pred, y_true, hparams)
        loss_buffer.append(loss.item())
        if len(loss_buffer) > 25:
            del loss_buffer[0]
        vis.update(loss.item(), hparams.learning_rate_init, step)
        print("Step %d   Avg. Loss %.4f   Loss %.4f" % 
              (step, np.mean(loss_buffer), loss.item()))
    
        # Backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Overwrite the latest version of the model
        if args.save_every != 0 and step % args.save_every == 0:
            vis.save()
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
            print("Current epoch: %d   Progress %.2f%%" % 
                  (dataset.epochs, dataset.epoch_progress * 100))