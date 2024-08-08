import math
import os
import time
from argparse import ArgumentParser
from tokengrams import MemmapIndex
import numpy as np
import plotly.graph_objects as go
import tempfile
import plotly.express as px

def benchmark_memmap(document_path: str, encoding_width=16):
    slice_sizes = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

    file_size = os.path.getsize(document_path)
    assert encoding_width % 8 == 0, "Encoding width must be a multiple of 8"
    total_tokens = file_size // (encoding_width / 8) # Divide by document word length in bytes

    times = []
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
        MemmapIndex.build(output_file, tmp_index_file, verbose=True)
        times.append(time.time() - start)
        print(f"Built index for slice of {size} tokens in {time.time() - start:.2f} seconds. Updated data:")
        print(times)

        os.remove(output_file)
        os.remove(tmp_index_file)

    return times

def kaleido_workaround():
    '''Write data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469'''
    with tempfile.NamedTemporaryFile() as temp_file:
        fig = px.scatter(x=[0], y=[0])
        fig.write_image(temp_file.name, format="pdf")
    time.sleep(2)

def plot(times_seconds: list[float]):
    kaleido_workaround()
    x = [10 ** i for i in range(len(times_seconds))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=times_seconds,
        mode='lines+markers',
        line=dict(shape='spline', smoothing=1.3, color='rgb(55, 126, 184)'),
        marker=dict(color='rgb(55, 126, 184)')
    ))

    fig.update_layout(
        title='Index Build Times over Corpus Sizes',
        xaxis_title='Corpus size (tokens)',
        yaxis_title='Build time (seconds)',
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

    y_min = 0.001
    y_max = 10_000
    y_tickvals = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000]

    fig.update_yaxes(
        type='log',
        range=[math.log10(y_min), math.log10(y_max)],
        tickmode='array',
        tickvals=y_tickvals,
        tickformat=","
    )

    fig.write_image("memmap_index_build_times.pdf", format='pdf')
    print("Plot saved to memmap_index_build_times.pdf")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="document.bin",
        help="Path to tokenized corpus file",
    )
    parser.add_argument(
        "--encoding_width",
        default=16,
        type=int,
        help="Number of bits to use for encoding each token"
    )
    parser.add_argument(
        "--benchmark_data",
        default = [
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
        ],
        type=list
    )
    args = parser.parse_args()

    if not args.benchmark_data:
        args.benchmark_data = benchmark_memmap(
            args.data_path,
            args.encoding_width
        )
    
    plot(args.benchmark_data)