from source_separation.visualizations import Visualizations
from source_separation.data_objects import MidiDataset
from source_separation.hparams import HParams
from source_separation.model import WavenetBasedModel, mse_loss, mae_loss, spectrogram_loss, mae_diff_loss
from pathlib import Path
import torch


def train(args, hparams: HParams):
    # Initialize visdom
    vis = Visualizations(args.run_name, averaging_window=25, auto_open_browser=True)
    
    # Create the datasets
    dataset = MidiDataset(
        root=args.dataset_root,
        is_train=True,
        chunk_size=int(args.chunk_duration * hparams.sample_rate),
        hparams=hparams,
    )
    data_iterator = dataset.generate(
        instruments=args.instruments,
        batch_size=args.batch_size,
        n_threads=4,
        max_chunks_per_music=args.max_chunks_per_music,
        chunk_reuse=args.chunk_reuse,
        chunk_pool_size=args.pool_size,
    )
    vis_dataset = MidiDataset(
        root=args.dataset_root,
        is_train=True,
        chunk_size=int(args.chunk_duration * hparams.sample_rate * args.batch_size),
        hparams=hparams,
    )
    vis_data_iterator = vis_dataset.generate(
        instruments=args.instruments,
        batch_size=1,
        n_threads=1,
        max_chunks_per_music=1,
        chunk_pool_size=1,
    )

    # Create the model and the optimizer
    model = WavenetBasedModel(len(args.instruments), hparams).cuda()
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
    for step, (x, y_true) in enumerate(data_iterator, init_step):
        # Forward pass and loss
        x, y_true = x.cuda(), y_true.cuda()
        y_pred = model(x)
        a, b, _ = mae_diff_loss(y_pred, y_true)
        loss = a + 0.2 * b
        print("MAE loss: %.3f   Diff loss: %.3f" % (a.item(), 0.2 * b.item()))

        # Visualizations
        vis.plot_loss(loss.item(), step)
    
        # Backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Overwrite the latest version of the model
        if args.save_every != -1 and step % args.save_every == 0:
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
        if args.vis_every != -1 and step % args.vis_every == 0:
            print("Creating visualizations, please wait... ", end="")
            x, y_true = next(vis_data_iterator)
            y_pred = model(x.cuda()).detach().cpu()
            vis.draw_waveform(y_pred.numpy()[0], y_true.numpy()[0], args.instruments)
            print("Done!")
            
            print(vis_dataset.debug_midi_fpaths)
    