#!/usr/bin/env python3
"""
Text-to-SFX Generator using AudioGen

This script generates sound effects from text descriptions using Meta's AudioGen model.

Usage:
    python text_to_sfx.py "dog barking in the distance"
    python text_to_sfx.py "thunder and rain" --duration 10 --output storm.wav
    python text_to_sfx.py "footsteps on gravel" -d 3 -o footsteps.wav

Examples of text prompts:
    - "dog barking"
    - "thunder and heavy rain"
    - "footsteps in a corridor"
    - "car engine starting"
    - "birds chirping in the morning"
    - "ocean waves crashing on rocks"
    - "fire crackling"
    - "crowd cheering at a stadium"
"""

import argparse
import sys
from pathlib import Path

import torch
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write


def generate_sfx(
    text: str,
    output_path: str = "output.wav",
    duration: float = 5.0,
    top_k: int = 250,
    top_p: float = 0.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
) -> Path:
    """
    Generate a sound effect from a text description.

    Args:
        text: Text description of the desired sound effect
        output_path: Output file path (extension will be added if not .wav)
        duration: Duration in seconds (default: 5.0)
        top_k: Top-k sampling (default: 250)
        top_p: Top-p (nucleus) sampling, 0 disables (default: 0.0)
        temperature: Sampling temperature (default: 1.0)
        cfg_coef: Classifier-free guidance coefficient (default: 3.0)

    Returns:
        Path to the generated audio file
    """
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Loading AudioGen model...")

    # Load the model
    model = AudioGen.get_pretrained("facebook/audiogen-medium")
    model.set_generation_params(
        duration=duration,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        cfg_coef=cfg_coef,
    )

    print(f"Generating {duration}s audio for: '{text}'")

    # Generate audio
    wav = model.generate([text], progress=True)

    # Prepare output path
    output = Path(output_path)
    stem = output.stem if output.suffix else str(output)

    # Save the audio
    audio_write(
        stem,
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )

    final_path = Path(f"{stem}.wav")
    print(f"Audio saved to: {final_path.absolute()}")

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate sound effects from text descriptions using AudioGen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s "dog barking"
    %(prog)s "thunder storm with heavy rain" --duration 10
    %(prog)s "footsteps on wooden floor" -d 3 -o footsteps.wav
    %(prog)s "car horn honking in traffic" --temperature 0.8

Prompt tips:
    - Be specific: "large dog barking aggressively" vs "dog barking"
    - Include environment: "footsteps in an empty hallway"
    - Mention qualities: "soft rain", "loud thunder", "distant sirens"
        """,
    )
    parser.add_argument("text", help="Text description of the sound effect to generate")
    parser.add_argument(
        "-o", "--output", default="output.wav", help="Output file path (default: output.wav)"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=5.0, help="Duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--top-k", type=int, default=250, help="Top-k sampling parameter (default: 250)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.0, help="Top-p (nucleus) sampling (default: 0.0, disabled)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--cfg-coef",
        type=float,
        default=3.0,
        help="Classifier-free guidance coefficient (default: 3.0)",
    )

    args = parser.parse_args()

    if not args.text.strip():
        print("Error: Please provide a non-empty text description", file=sys.stderr)
        sys.exit(1)

    try:
        generate_sfx(
            text=args.text,
            output_path=args.output,
            duration=args.duration,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            cfg_coef=args.cfg_coef,
        )
    except Exception as e:
        print(f"Error generating audio: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
