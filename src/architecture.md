``` rust
struct SuffixTable {}
impl SuffixTable {
    pub fn count_next(&self);

    /// To be deleted - not a natural fit for a suffix table data structure, and depends on count_next
    /// which is "overridden" in ShardedMemmapIndex. I want to move it into a sampler struct.
    pub fn sample(&self) {
        counts = self.count_next()
        do_sample(counts)
    }
}

struct MemmapIndex {
    table: SuffixTable
}
impl MemmapIndex {
    ...lots of index/table creation code

    pub fn count_next(&self) {
        self.table.count_next()
    }

    pub fn sample(&self) {
        self.table.sample()
    }
}

struct ShardedInMemoryIndex {
    shards: Vec<MemmapIndex>
}
impl ShardedMemmapIndex {
    pub fn count_next(&self) {
        counts = self.shards.iter().map(|shard| shard.count_next())
        merge_counts(counts)
    }
}

// WIP
struct Sampler {
    // Not possible due to PyO3 not supporting lifetimes
    index: Box<Index>
}
impl Sampler {
    pub fn sample(&self) {
        counts = self.index.count_next()
        do_sample(counts)
    }
}
```