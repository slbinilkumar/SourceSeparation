from source_separation.data_objects import MidiDataset
from source_separation.model import WavenetBasedModel
from source_separation.model import mae_loss, spectrogram_loss, mse_loss, mae_diff_loss
from pathlib import Path
import numpy as np
import torch


def test(args, hparams):
    dataset = MidiDataset(
        root=args.dataset_root,
        is_train=False,
        hparams=hparams,
        chunk_size=int(args.chunk_duration * hparams.sample_rate)
    )
    
    dataloader = dataset.generate(
        instruments=args.instruments,
        batch_size=args.batch_size,
        n_threads=4,
        chunk_pool_size=1,
    )

    # Load the model
    model = WavenetBasedModel(len(args.instruments), hparams).cuda()
    state_fpath = Path("saved_models", "%s.pt" % args.run_name)
    init_step = model.load(state_fpath, None, args.instruments, hparams)
    print("Loaded model %s from step %d." % (state_fpath, init_step))
        
    # Set the model to eval mode
    model.eval()
    
    losses = []
    loss_names = ["MAE", "MSE", "Spec", "Diff"]
    print("Computing the loss on the test set:")
    for step, (x, y_true) in enumerate(dataloader):
        with torch.no_grad():
            x, y_true = x.cuda(), y_true.cuda()

            # Forward the batch
            y_pred = model(x)
            mae = mae_loss(y_pred, y_true)
            mse = mse_loss(y_pred, y_true)
            spec = spectrogram_loss(y_pred, y_true, hparams)
            mae_diff1, mae_diff2, _ = mae_diff_loss(y_pred, y_true)
            mae_diff = mae_diff1 + 0.05 * mae_diff2

            losses.append([mae, mse, spec, mae_diff])
            print("   Step: %4d:")
            for loss, loss_name in zip(np.mean(losses, axis=0), loss_names):
                print("%-4s: %.4f" % loss)