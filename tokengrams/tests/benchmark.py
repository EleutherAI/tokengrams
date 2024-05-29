import numpy as np
import os
import csv
from pathlib import Path

from tokengrams import MemmapIndex
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    # bus error at 1_000_000
    seq_counts = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    table_path = Path.cwd() / 'table.idx'
    pile_path = Path('/') / 'mnt' / 'ssd-1' / 'pile_preshuffled' / 'standard' / 'document.bin'
    num_elements = os.path.getsize(pile_path) // np.dtype(np.uint16).itemsize
    pile_mmap = np.memmap(pile_path, dtype=np.uint16, mode="r", shape=(num_elements // 2049, 2049))

    rayon_times = []
    sushi_times = []

    def run_sushi_build(seq_count):
        thread_sample_path = f'sample-{seq_count}.bin'
        sample_mmap = np.memmap(thread_sample_path, dtype=np.uint16, mode="w+", shape=(seq_count, 2049))
        sample_mmap[:] = pile_mmap[:seq_count]
        sample_mmap.flush()
        del sample_mmap

        start_time = time.time()
        MemmapIndex.build_sushi(str(thread_sample_path), str(table_path), verbose=True)
        build_time = time.time() - start_time
        return build_time


    with ThreadPoolExecutor() as executor:
        sushi_futures = {executor.submit(run_sushi_build, count): count for count in seq_counts}

        for future in as_completed(sushi_futures):
            sushi_time = future.result()
            sushi_times.append(sushi_time)
            with open('sushi-output.csv', 'w', newline='') as file:
                csv.writer(file).writerow(sushi_times)

    sample_path = Path.cwd() / 'sample-rayon.bin'
    for seq_count in seq_counts:
        sample_mmap = np.memmap(sample_path, dtype=np.uint16, mode="w+", shape=(seq_count, 2049))
        sample_mmap[:] = pile_mmap[:seq_count]
        sample_mmap.flush()
        del sample_mmap

        start = time.time()
        MemmapIndex.build(str(sample_path), str(table_path), verbose=True)
        rayon_times.append(time.time() - start)
        with open('rayon-output.csv', 'w', newline='') as file:
            csv.writer(file).writerow(rayon_times)


    pd.DataFrame({
        'sequence': seq_counts,
        'rayon': rayon_times,
        'sushi': sushi_times
    }).to_csv('benchmark.csv', index=False)


if __name__ == "__main__":
    main()
