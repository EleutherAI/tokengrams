use funty::Unsigned;
use memmap2::{Mmap, MmapAsRawDesc, MmapMut};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// An immutable memory-mapped slice of unsigned integers
pub struct MmapSlice<T: Unsigned> {
    mmap: Mmap,
    _element_type: PhantomData<T>,
}

impl<T: Unsigned> MmapSlice<T> {
    pub fn new<F: MmapAsRawDesc>(file: F) -> std::io::Result<Self> {
        let raw = unsafe { Mmap::map(file)? };

        // Sanity check that the file size is a multiple of the element size.
        if raw.len() % std::mem::size_of::<T>() != 0 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File size is not a multiple of element size",
            ))
        } else {
            Ok(MmapSlice {
                mmap: raw,
                _element_type: PhantomData,
            })
        }
    }

    // Return the number of items of type T that can fit in the memory map.
    pub fn len(&self) -> usize {
        self.mmap.len() / std::mem::size_of::<T>()
    }

    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr() as *const T, self.len()) }
    }
}

impl<T: Unsigned> Deref for MmapSlice<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

/// A mutable memory-mapped slice of unsigned integers
pub struct MmapSliceMut<T: Unsigned> {
    mmap: MmapMut,
    _element_type: PhantomData<T>,
}

impl<T: Unsigned> MmapSliceMut<T> {
    pub fn new<F: MmapAsRawDesc>(file: F) -> std::io::Result<Self> {
        let raw = unsafe { MmapMut::map_mut(file)? };

        Ok(MmapSliceMut {
            mmap: raw,
            _element_type: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.mmap.len() / std::mem::size_of::<T>()
    }

    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr() as *const T, self.len()) }
    }

    pub fn as_slice_mut<'a>(&'a mut self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.mmap.as_mut_ptr() as *mut T, self.len()) }
    }

    pub fn into_read_only(self) -> std::io::Result<MmapSlice<T>> {
        Ok(MmapSlice {
            mmap: self.mmap.make_read_only()?,
            _element_type: PhantomData,
        })
    }
}

impl<T: Unsigned> Deref for MmapSliceMut<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Unsigned> DerefMut for MmapSliceMut<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}
