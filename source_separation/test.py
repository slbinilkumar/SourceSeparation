from source_separation.data_objects import MidiDataset, get_instrument_name
from source_separation.model import Model, spectrogram_loss
from time import perf_counter as timer
from pathlib import Path
import sounddevice as sd
import numpy as np
import torch


def test(args, hparams):
    dataset = MidiDataset(
        root=args.dataset_root,
        is_train=False,
        hparams=hparams,
    )
    
    dataloader = dataset.generate(
        instruments=args.instruments,
        batch_size=1,
        n_threads=2,
        chunk_pool_size=1,
    )

    # Load the model
    model = Model(len(args.instruments), hparams).cuda()
    state_fpath = Path("saved_models", "%s.pt" % args.run_name)
    init_step = model.load(state_fpath, None, args.instruments, hparams)
    print("Loaded model %s from step %d." % (state_fpath, init_step))
        
    # Set the model to eval mode
    model.eval()
    
    print("\nPlaying clips one by one, press any key to go to the next one.\n")
    x = torch.from_numpy(np.load("x.npy"))
    y_true = torch.from_numpy(np.load("y_true.npy"))
    while True:
    # for x, y_true in dataloader:
        with torch.no_grad():
            x, y_true = x.cuda(), y_true.cuda()

            # Forward the batch
            start = timer()
            y_pred = model(x)
            print("\nForwarded %d seconds of audio in %dms." % 
                  (args.chunk_duration, round((timer() - start) * 1000)))
            
            ## Compute the loss
            # loss = spectrogram_loss(y_pred, y_true, hparams)
            # print("Loss %.4f" % loss.item())
            
            # Process the waveforms for playing
            x = x.squeeze().cpu().numpy()
            y_pred = y_pred.squeeze(0).cpu().numpy()
            y_pred = np.clip(y_pred, -1, 1)
            
            # Listen to the audio waveforms
            print("Playing the source audio")
            sd.play(x, samplerate=args.sample_rate, blocking=True)
            
            for yi_pred, instrument_id in zip(y_pred, args.instruments): 
                print("Playing the extracted %s" % get_instrument_name(instrument_id))
                sd.play(yi_pred, samplerate=args.sample_rate, blocking=True)
                