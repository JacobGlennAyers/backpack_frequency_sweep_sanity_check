# backpack_frequency_sweep_sanity_check


qa_sweep.py
-----------
Streamlined single-script QA tool for bird-backpack FM radio transmitters.

Three subcommands:

  generate   — write a log-sweep WAV to disk
  play       — play it through the speaker
  analyze    — compare a received WAV against the sweep and report bandwidth

Typical session:

  python qa_sweep.py generate
  
  python qa_sweep.py play
  
  # ... place backpack on speaker, start recording on receiver, run play, stop recorder ...
  
  python qa_sweep.py analyze received.wav

