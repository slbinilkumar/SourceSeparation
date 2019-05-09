from source_separation.data_objects import MidiDataset, get_instrument_name
from source_separation.model import WavenetBasedModel
from pathlib import Path
import numpy as np
import librosa
import torch


def generate(args, hparams):
    dataset = MidiDataset(None, None, int(args.chunk_duration * hparams.sample_rate), hparams)
    args.out_dir.mkdir(exist_ok=True)
    
    # Load the model
    model = WavenetBasedModel(len(args.instruments), hparams).cuda()
    state_fpath = Path("saved_models", "%s.pt" % args.run_name)
    init_step = model.load(state_fpath, None, args.instruments, hparams)
    print("Loaded model %s from step %d." % (state_fpath, init_step))
    
    # Set the model to eval mode
    model.eval()
    
    # Forward chunks
    chunks = dataset.extract_chunks(args.midi_fpath, args.instruments, shuffled=False)
    print("Instruments: %s" % chunks[0][2])
    pred_chunks = []
    gt_chunks = []
    duration = 0
    for step, chunk in enumerate(chunks):
        with torch.no_grad():
            print("Chunk %d" % step, end="")
            x, y_true = dataset.collate([chunk], args.instruments)
            gt_chunks.append(y_true.numpy().squeeze(0))
            x, y_true = x.cuda(), y_true.cuda()
            y_pred = model(x)
            
            y_pred = y_pred.cpu().numpy().squeeze(0)
            pred_chunks.append(y_pred)
            duration += y_pred.shape[1] / hparams.sample_rate
            print(" %.3f" % duration)
            if duration >= args.music_duration:
                break
                
    # Crossfade chunks:
    if len(pred_chunks) > 1:
        raise NotImplemented()
    else:
        pred_tracks = pred_chunks[0]
        gt_tracks = gt_chunks[0]

    # Save the wavs
    for y, mode in zip([pred_tracks, gt_tracks], ["pred", "gt"]):
        for track, instrument_id in zip(y, args.instruments):
            instrument_name = get_instrument_name(instrument_id)
            fname = "_".join((args.midi_fpath.stem, args.run_name, instrument_name, mode)) + ".mp3"
            fpath = args.out_dir.joinpath(fname)
            librosa.output.write_wav(fpath, track, hparams.sample_rate)
            