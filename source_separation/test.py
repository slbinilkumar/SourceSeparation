from source_separation.data_objects import MidiDataset
from source_separation.model import Model
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
        source_instruments=args.source_instruments,
        target_instruments=args.target_instruments,
        batch_size=1,
        n_threads=2,
        chunk_pool_size=1,
    )

    # Load the model
    model = Model(hparams).cuda()
    state_fpath = Path("model.pt")
    checkpoint = torch.load(state_fpath)
    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model_state"])
    print("Loaded model %s from step %d." % (state_fpath, step))
        
    # Set the model to eval mode
    model.eval()
    
    # If you want to have a look at the data
    for batch in dataloader:
        with torch.no_grad():
            x, y_true = torch.from_numpy(batch).cuda()
            
            # Forward the batch
            start = timer()
            y_pred = model(x)
            torch.cuda.synchronize()
            delta = timer() - start
            print("Forwarded %d seconds of audio in %dms." % 
                  (args.chunk_duration, round(delta * 1000)))
            
            # Compute the loss
            loss = model.loss(y_pred, y_true)
            print("Loss %.4f" % loss.item())
            
            # Process the waveforms for playing
            x = x.squeeze().cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            y_pred = (y_pred / y_pred.max()) * x.max()
            
            # Listen to the audio waveforms
            print("Playing the source audio:")
            sd.play(x, samplerate=args.sample_rate, blocking=True)
            print("Playing the generated audio:")
            sd.play(y_pred, samplerate=args.sample_rate, blocking=True)
        