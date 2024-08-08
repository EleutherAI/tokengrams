import math
import os
import time
from argparse import ArgumentParser
from typing import Literal
from pathlib import Path

from tokengrams import MemmapIndex, InMemoryIndex
import numpy as np
import plotly.graph_objects as go

def benchmark(document_path: str, cls: Literal["InMemoryIndex", "MemmapIndex"], encoding_width=16, vocab=2**16):
    slice_sizes = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

    file_size = os.path.getsize(document_path)
    assert encoding_width % 8 == 0, "Encoding width must be a multiple of 8"
    total_tokens = file_size // (encoding_width / 8) # Divide by document word length in bytes

    build_times = []
    count_next_times = []

    tokens = np.memmap(document_path, dtype=np.uint16, mode='r')
    for size in slice_sizes:
        if size > total_tokens:
            print(f"Skipping slice size {size} as it exceeds the total number of tokens.")
            continue
        
        slice_data = tokens[:size]
        output_file = f"tmp_slice_{size}.bin"
        with open(output_file, 'wb') as f:
            slice_data.tofile(f)
        
        print(f"Created file from slice of {size} tokens: {output_file}")
        
        # Build index
        tmp_index_file = f"tmp_slice_{size}.idx"
        start = time.time()
        if cls == "MemmapIndex":
            index = MemmapIndex.build(output_file, tmp_index_file, verbose=True)
        else:
            index = InMemoryIndex.from_token_file(output_file, verbose=True)
        build_times.append(time.time() - start)
        print(f"Built index for slice of {size} tokens in {time.time() - start:.2f} seconds. Updated data:")
        print(build_times)

        # Count next with empty query (count unigrams)
        start = time.time()
        index.count_next([])
        count_next_times.append(time.time() - start)

        os.remove(output_file)
        if os.path.exists(tmp_index_file):
            os.remove(tmp_index_file)

    return build_times, count_next_times

def plot(
    times: list[float], 
    cls: Literal["InMemoryIndex", "MemmapIndex"],
    label: Literal["build", "count_next"]
) -> None:
    x = [10 ** i for i in range(len(times))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=times,
        mode='lines+markers',
        line=dict(shape='spline', smoothing=1.3, color='rgb(55, 126, 184)'),
        marker=dict(color='rgb(55, 126, 184)')
    ))

    fig.update_layout(
        title=f'{cls} {label} times over corpus sizes',
        xaxis_title='Corpus size (tokens)',
        yaxis_title=f'{label.capitalize()} time (seconds)',
        width=800,
        height=500,
        margin=dict(l=80, r=50, t=80, b=80)
    )

    ticktext = [f'1e{int(math.log10(val))}' for val in x]
    fig.update_xaxes(
        type='log',
        tickmode='array',
        tickvals=x,
        ticktext=ticktext
    )

    # Get the enclosing powers of ten for the range
    y_log10_floor = math.floor(math.log10(min(times)))
    y_log10_ceil = math.ceil(math.log10(max(times))) 
    
    y_tickvals = np.logspace(y_log10_floor, y_log10_ceil, num=(y_log10_ceil - y_log10_floor) + 1)

    fig.update_yaxes(
        type='log',
        range=[y_log10_floor, y_log10_ceil],
        tickmode='array',
        tickvals=y_tickvals,
        tickformat=","
    )

    output_path = Path(f"tokengrams/benchmark/{cls}_{label}_times.png")
    fig.write_image(output_path, scale=5)
    print(f"Plot saved to {str(output_path)}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", default=None, help="Path to tokenized corpus file")
    parser.add_argument("--encoding_width", default=16, type=int, help="Bits per token")
    parser.add_argument("--cls", default="MemmapIndex", choices=["InMemoryIndex", "MemmapIndex"], 
        help="Index class to benchmark")
    args = parser.parse_args()

    if args.data_path:
        build_times, count_next_times = benchmark(
            args.data_path,
            args.cls,
            args.encoding_width
        )
        plot(build_times, args.cls, label="build")
        plot(count_next_times, args.cls, label="count_next")
    else:
        print("No path to token corpus found, plotting precomputed benchmark data for MemmapIndex:")
        benchmark_data = [
            0.00757908821105957, 
            0.0053365230560302734, 
            0.0067026615142822266, 
            0.009454727172851562, 
            0.026259660720825195, 
            0.11438727378845215, 
            1.1938815116882324, 
            10.36899209022522, 
            110.39609742164612, 
            1263.0914874076843
        ]
        plot(benchmark_data, "MemmapIndex", label="build")