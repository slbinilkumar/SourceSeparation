from source_separation.visualizations import Visualizations
from source_separation.data_objects import MidiDataset
from source_separation.hparams import HParams
from source_separation.model import Model, spectrogram_loss, mae_loss, mse_loss, CombinedLoss
from pathlib import Path
import numpy as np
import torch


def train(args, hparams: HParams):
    # Initialize visdom
    vis = Visualizations(args.run_name, averaging_window=25, auto_open_browser=True)
    
    # Create the datasets
    dataset = MidiDataset(
        root=args.dataset_root,
        is_train=True,
        chunk_size=hparams.chunk_size,
        hparams=hparams,
    )
    data_iterator = dataset.generate(
        instruments=args.instruments,
        batch_size=args.batch_size,
        n_threads=4,
        chunk_reuse=args.chunk_reuse,
        chunk_pool_size=args.pool_size,
    )
    vis_dataset = MidiDataset(
        root=args.dataset_root,
        is_train=True,
        chunk_size=10 * hparams.sample_rate,
        hparams=hparams,
    )
    vis_data_iterator = vis_dataset.generate(
        instruments=args.instruments,
        batch_size=1,
        n_threads=1,
        chunk_pool_size=1,
    )

    # Create the model and the optimizer
    model = Model(len(args.instruments), hparams).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hparams.learning_rate_init, 
    )

    # Load any existing model
    Path("saved_models").mkdir(exist_ok=True)
    state_fpath = Path("saved_models", "%s.pt" % args.run_name)
    if state_fpath.exists():
        print("Found existing model \"%s\", loading it and resuming training." % state_fpath)
        init_step = model.load(state_fpath, optimizer, args.instruments, hparams)
    else:
        print("No model \"%s\" found, starting training from scratch." % state_fpath)
        init_step = 1

    # Set the model to training mode
    model.train()
    
    # Setup the visualizations environment
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_params(args.__dict__, "Arguments")
    vis.log_params(hparams.__dict__, "Hyperparameters")
    vis.log_implementation({"Device": device_name})

    # Training loop
    loss_fn = CombinedLoss(hparams, 500, 5000, 100)
    loss_buffer = []
    for step, (x, y_true) in enumerate(data_iterator, init_step):
        # Forward pass and loss
        x, y_true = x.cuda(), y_true.cuda()
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true, step)
        
        # Visualizations
        loss_buffer.append(loss.item())
        if len(loss_buffer) > 25:
            del loss_buffer[0]
        vis.update(loss.item(), hparams.learning_rate_init, step)
        # print("Step %d   Avg. Loss %.4f   Loss %.4f" % 
        #       (step, np.mean(loss_buffer), loss.item()))
    
        # Backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Overwrite the latest version of the model
        if args.save_every != 0 and step % args.save_every == 0:
            print("Saving the model (step %d)" % step)
            vis.save()
            torch.save({
                "instruments": args.instruments,
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
            print("Current epoch: %d   Progress %.2f%%" % 
                  (dataset.epochs, dataset.epoch_progress * 100))
            
        # Draw the generated audio waveforms and plot them for comparison
        if args.vis_every != 0 and step % args.vis_every == 0:
            print("Creating visualizations, please wait... ", end="")
            x, y_true = next(vis_data_iterator)
            y_pred = model(x.cuda()).detach().cpu()
            vis.draw_waveform(y_pred.numpy()[0], y_true.numpy()[0], args.instruments)
            print("Done!")
